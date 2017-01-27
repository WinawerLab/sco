# metamers.py
#
# code specifically for evaluating SCO model performance on "texture metamers", as discussed in
# Freeman2013 and Portilla2000. See Metamers.ipynb for more information.
#
# References:
# - Freeman, J., Ziemba, C. M., Heeger, D. J., Simoncelli, E. P., & Movshon, J. A. (2013). A
#   functional and perceptual signature of the second visual area in primates. Nature Neuroscience.
# - Portilla, J., & Simoncelli, E. P. (2000). A parametric texture model based on joint statistics of
#   complex wavelet coefficients. International journal of computer vision.
# 
# by William F. Broderick

import re

def search_for_mets(x):
    tmp = re.search(r'(V[12]Met(Scaled)?).*', x)
    if tmp is None:
        return "original"
    else:
        return tmp.groups()[0].replace('MetScaled', 'SclMet').replace('Met', '-metamer')

def search_for_noise_seed(x):
    tmp = re.search(r'im[0-9]+-smp1-([0-9]+).*png', x)
    if tmp is None:
        return None
    else:
        return tmp.groups()[0]

def create_met_df(df, col_name='image'):
    """for each image, determine its type, name, and seed
    """
    if 'language' in df.columns:
        df = df[df.language=='python']
    df['image_type'] = df[col_name].apply(search_for_mets)
    df['image_name'] = df[col_name].apply(lambda x: re.search(r'(im[0-9]+)-smp1.*png', x).groups()[0])
    df['image_seed'] = df[col_name].apply(search_for_noise_seed)
    return df
