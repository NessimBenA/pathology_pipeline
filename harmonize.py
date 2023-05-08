import numpy as np # import cupy for gpu
import os
import shutil

from joblib import delayed, Parallel
from skimage.exposure import match_histograms # from cusim.skimage.exposure for gpu
from skimage.color import rgb2hed, hed2rgb # from cusim.skimage.color for gpu
from skimage.io import imread, imsave



def copy_image(src, dst, transform=None):
    '''
    Copy image from src to dst and apply the choosen transform (if there is one).

        ### Parameters

            src: str
                path to source image
            dst: str
                path to new image
            transform: function, default: None
                function that takes an image and returns another
    '''
    if transform is None:
        shutil.copy(src, dst)
    else:
        imsave(dst, transform(imread(src)))



def copy_images(srcs, dsts, transform=None):
    '''
    Copy images from src to dst and apply the choosen transform (if there is one).

        ### Parameters

            srcs: iterable of str
                paths to source image
            dsts: iterable of str
                paths to new image
            transform: function, default: None
                function that takes an image and returns another
    '''
    for src, dst in zip(srcs, dsts): copy_image(src, dst, transform)



try:
    def get_ihc_hde(img):
        '''
            Convert rgb image to hed, then make a new rgb image for each axis (h, e, d).

            ### Parameters

                img: ndarray of integers
                    rgb image to harmonize
            
            ### Parameters

                ihc_h: ndarray of integers
                    rgb image from h axis
                ihc_e: ndarray of integers
                    rgb image from e axis
                ihc_d: ndarray of integers
                    rgb image from d axis
            
            References
            
               [1] A. C. Ruifrok and D. A. Johnston, "Quantification of histochemical
                staining by color deconvolution.," Analytical and quantitative
                cytology and histology / the International Academy of Cytology [and]
                American Society of Cytology, vol. 23, no. 4, pp. 291-9, Aug. 2001.
        '''
        ihc_hed=(rgb2hed(img))
        null = np.zeros_like(ihc_hed[:, :, 0])
        ihc_h = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
        ihc_e = hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
        ihc_d = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))
        return ihc_h, ihc_e, ihc_d

    import config as cfg
    IHC_H_REF, IHC_E_RED, IHC_D_REF = get_ihc_hde(imread(cfg.REF_IMAGE))

    def harmonizer(img):
        '''
            Harmonize the image through hed conversion, erase the color and keep the stains.
            Then match the histogram with a reference image, to avoid classification biases because of coloration.
            You must define an hed image as HED_REF, it will be the reference for harmonization.

            ### Parameters

                img: ndarray of integers
                    rgb image to harmonize
            
            References
            
               [1] A. C. Ruifrok and D. A. Johnston, "Quantification of histochemical
                staining by color deconvolution.," Analytical and quantitative
                cytology and histology / the International Academy of Cytology [and]
                American Society of Cytology, vol. 23, no. 4, pp. 291-9, Aug. 2001.
        '''
        ihc_h, ihc_e, ihc_d = get_ihc_hde(img)
        return (np.dstack((match_histograms(ihc_h, IHC_H_REF)[:, :, 0],
                        match_histograms(ihc_d, IHC_E_RED)[:, :, 1],
                        match_histograms(ihc_e, IHC_D_REF)[:, :, 2])
                        )*255).astype(np.uint8)

except FileNotFoundError:
    print("WARNING: To use the harmonizer, please define a reference image as HED_REF (l.83)")
    pass



if __name__ == '__main__':
    import timeit

    start = timeit.time.time_ns()

    # Find data and prepare paths
    filenames = [filename for filename in os.listdir(cfg.SLICES_PATH) 
                if filename.endswith(cfg.SLICE_EXT)]
    srcpaths = np.array([os.path.join(cfg.SLICES_PATH, filename) for filename in filenames])
    dstpaths = np.array([os.path.join(cfg.HARMONIZED_SLICES_PATH, filename) for filename in filenames])
    
    # Batch paths with used cpus
    batch_size = int(srcpaths.shape[0]/cfg.USED_CPU)
    srcbatches = [srcpaths[x:x+batch_size] for x in range(0, srcpaths.shape[0], batch_size)]
    dstbatches = [dstpaths[x:x+batch_size] for x in range(0, dstpaths.shape[0], batch_size)]
    del srcpaths, dstpaths, filenames

    # copy each piece and transform them if needed
    Parallel(n_jobs=cfg.USED_CPU)(delayed(copy_images)(
        srcs = srcs, 
        dsts = dsts, 
        transform = harmonizer)
        for srcs, dsts in zip(srcbatches, dstbatches))
    
    print(f"{(timeit.time.time_ns() - start)/10e8} seconds taken")