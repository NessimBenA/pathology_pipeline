import cv2 as cv
import math
import numpy as np
import os

from joblib import Parallel, delayed
from openslide import OpenSlide
from PIL import Image
from skimage.io import imsave
from tqdm import tqdm

def gpu_computation(activate=False):
    '''
        Enable or disable gpu computation.
        To enable it, cupy and cucim modules are needed.

        ### Parameters

            activate: boolean, default: False
                activate or not gpu computation
    '''
    global bapdmskimg, bapdmcp # strange names to avoid conflicts
    if activate == True:
        import cucim.skimage as bapdmskimg
        import cupy as bapdmcp
    else:
        import skimage as bapdmskimg
        import numpy as bapdmcp

Image.MAX_IMAGE_PIXELS = None
gpu_computation()



def select_device(device):
    '''
        Select device used for some gpu computations of the current process.

        ### Parameters

            device: int or None
                selected gpu (if None, does nothing)
    '''
    if device is not None:
        try:
            bapdmcp.cuda.Device(device).use()
        except AttributeError:
            print("using cpu, for gpu computation please call gpu_computation(activate=True)")



def cupy_to_numpy(arr):
    '''
        Return a numpy ndarray from a cupy ndarray.

        ### Parameters

            arr: ndarray (from cupy or numpy)
                array to convert as numpy one (if not already)
    '''
    try:
        return arr.get() # if using gpu (cupy)
    except AttributeError:
        return arr



def get_preview(slide, xresolution, yresolution):
    '''
    Read the slide and returns its preview (resolution=1)

        ### Parameters

            slide: openslide.OpenSlide
                slide to read
            xresolution: int
                aimed xresolution for segmentation
            yresolution: int
                aimed yresolution for segmentation
        
        ### Returns

            preview: numpy.array
                fully loaded slide with resolution=1
    '''
    width, height = slide.dimensions
    new_width = int(width/xresolution)
    new_height = int(height/yresolution)
    
    amount = math.ceil(xresolution/2) # split into same sized tiles
    w = int(width/amount) # tile width
    h = int(height/amount) # tile height
    nw = int(new_width/amount) # tile new width
    nh = int(new_height/amount) # tile new height

    preview = bapdmcp.zeros((new_height, new_width, 3), dtype=bapdmcp.uint8) + 255
    for i in range(amount):
        for j in range(amount):
            preview[j*nh:(j+1)*nh, i*nw:(i+1)*nw] = bapdmcp.array(cv.resize(
                np.array(slide.read_region((i*w,j*h), 0, (w, h)), dtype=np.uint8)[:,:,:3], 
                dsize=(nw, nh), interpolation=cv.INTER_NEAREST))
    return preview



def get_cleaned_binary(preview, fp_val=16, sigma=2):
    '''
    Read the slide and returns it with the new resolution.

        ### Parameters

            preview: uint8 ndarray
                fully loaded slide with resolution=1
            fpval: int, default: 16
                value that is used to creates footprints for segmentation
            sigma: float, default: 2
                sigma value for gaussian filter (applied on the slide before segmentation)

        ### Returns

            bw: ndarray of numpy.bool_
                binary image from the preview (slide with resolution=1)   
    '''
    bw = bapdmskimg.color.rgb2gray(preview)
    bw = bapdmskimg.filters.threshold_otsu(bw) > bapdmskimg.filters.gaussian(bw, sigma)
    bw = bapdmskimg.morphology.opening(bw, bapdmskimg.morphology.square(fp_val))
    bw += bapdmskimg.segmentation.clear_border(~bw)
    bw = bapdmskimg.morphology.dilation(bw, bapdmskimg.morphology.disk(int(fp_val*0.25)))
    return bw



def mask_rgb(rgb, mask):
    '''
    Mask rgb image with a binary image.

        ### Parameters

            rgb: ndarray of numpy.uint8
                rgb image
            mask: ndarray of numpy.bool_
                binary image

        ### Returns

            masked_rgb: ndarray of numpy.uint8
                rgb image masked with the binary image
    '''

    mask_rgb = bapdmcp.repeat(mask[...,None],3,axis=2)
    return mask_rgb*rgb + (~mask_rgb*255).astype(bapdmcp.uint8)



def save_segmentation(basepath, slide, bw, xresolution, yresolution, ext="png"):
    '''
    Read the slide and save its segmentations from a binary image.

        ### Parameters

            basepath: str
                slice path, ends with _{slice_number}
            slide: openslide.OpenSlide
                slide to read
            bw: ndarray of numpy.bool_
                binary image from the preview (slide with resolution=1)
            xresolution: int
                aimed xresolution for segmentation
            yresolution: int
                aimed yresolution for segmentation
            ext: str, default: "png"
                slices extension 
    '''
    for i, region in enumerate(bapdmskimg.measure.regionprops(bapdmskimg.measure.label(bw))):
        x, y = region.bbox[:2]
        w, h = region.bbox[2] - x, region.bbox[3] - y
        if w*h > 10000:
            x *= xresolution
            y *= yresolution
            w *= xresolution
            h *= yresolution
            bw = bapdmskimg.transform.resize(region.image, (w,h))
            imsave(f"{basepath}_{i}.{ext}", mask_rgb(rgb=np.array(slide.read_region(
                (y,x), 0, (h, w)), dtype=np.uint8)[:,:,:3], mask=cupy_to_numpy(bw)))



def save_preview(path, slide, xresolution, yresolution):
    '''
    Read the slide and save its preview (resolution=1)

        ### Parameters

            path: str
                path to save the preview
            slide: openslide.OpenSlide
                slide to read
            xresolution: int
                aimed xresolution for segmentation
            yresolution: int
                aimed yresolution for segmentation
    '''
    preview = get_preview(slide, xresolution, yresolution)
    if preview is not None: imsave(path, preview)



def save_slices_from_slide(slidepath, basepath, xresolution, yresolution, fpval=16, sigma=2, ext="png", device=None):
    '''
    Read the slide segment to get slices and save them with xresolution and yresolution.

        ### Parameters

            slidepath: str
                path to the slide
            basepath: str
                slice path, ends with _{slice_number}
            xresolution: int
                aimed xresolution for segmentation
            yresolution: int
                aimed yresolution for segmentation
            fpval: int, default: 16
                value that is used to creates footprints for segmentation
            sigma: float, default: 2
                sigma value for gaussian filter (applied on the slide before segmentation)
            ext: str, default: "png"
                slices extension 
            device: int, default: None
                selected gpu (if None, no gpu used)
    '''
    select_device(device)
    slide = OpenSlide(slidepath)
    preview = get_preview(slide, xresolution, yresolution)
    bw = get_cleaned_binary(preview, fpval, sigma)
    save_segmentation(basepath, slide, bw, xresolution, yresolution, ext)



def save_slices_from_preview(slidepath, basepath, xresolution, yresolution, fpval=16, sigma=2, ext="png", device=None):
    '''
    Read the slide segment to get slices and save them with xresolution and yresolution.

        ### Parameters

            slidepath: str
                path to the slide
            basepath: str
                slice path, ends with _{slice_number}
            xresolution: int
                slide xresolution
            yresolution: int
                slide yresolution
            fpval: int, default: 16
                value that is used to creates footprints for segmentation
            sigma: float, default: 2
                sigma value for gaussian filter (applied on the slide before segmentation)
            ext: str, default: "png"
                slices extension
            device: int, default: None
                selected gpu (if None, no gpu used)
    '''
    select_device(device)
    preview = get_preview(OpenSlide(slidepath), xresolution, yresolution)
    bw = get_cleaned_binary(preview, fpval, sigma)
    for i, region in enumerate(bapdmskimg.measure.regionprops(bapdmskimg.measure.label(bw))):
        if (region.bbox[2]-region.bbox[0]) * (region.bbox[3]-region.bbox[1]) > 10000:
            imsave(fname = f"{basepath}_{i}.{ext}", arr = cupy_to_numpy(mask_rgb(
              rgb = preview[region.bbox[0]:region.bbox[2], region.bbox[1]:region.bbox[3]], 
              mask = region.image)))



if __name__ == '__main__':
    import config as cfg
    import timeit
    if cfg.USED_GPU > 0: gpu_computation(True)

    start = timeit.time.time_ns()
    filenames = [filename for filename in os.listdir(cfg.SLIDES_PATH)
      if filename.endswith(cfg.SLIDE_EXT)]
    
    Parallel(n_jobs=cfg.USED_CPU)(delayed(save_slices_from_preview)(
      slidepath = os.path.join(cfg.SLIDES_PATH, filename),
      basepath = os.path.join(cfg.SLICES_PATH, filename[:-(len(cfg.SLIDE_EXT)+1)]),
      xresolution = cfg.SLICE_RESOLUTION,
      yresolution = cfg.SLICE_RESOLUTION,
      fpval = cfg.FP_VALUE,
      sigma = cfg.BLUR_SIGMA,
      ext = cfg.SLICE_EXT,
      device = None if cfg.USED_GPU <= 0 else i%cfg.USED_GPU)
      for i, filename in tqdm(enumerate(filenames), total=len(filenames)))
    
    print(f"{(timeit.time.time_ns() - start)/10e8} seconds taken")
