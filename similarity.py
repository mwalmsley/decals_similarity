import pickle
import logging

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.units as u

@st.cache
def load_catalog():
    return pd.read_parquet('/home/walml/repos/astronomaly/dr5_dr8_catalog_with_radius.parquet').reset_index(drop=True)

@st.cache
def load_features():
    return pd.read_parquet('/home/walml/repos/astronomaly/dr5_8_b0_pca10_and_safe_filenames.parquet')


# can't cache as weird datatype
def catalog_to_coordinates(df):
    return SkyCoord(list(df['ra']), list(df['dec']), unit='deg')


def get_url(galaxy, size=250):
    radius = galaxy['estimated_radius']
    ra = galaxy['ra']
    dec = galaxy['dec']
    # partial duplicate of downloader.get_download_url, but for png and fixed size rather than native scale
    pixscale = max(radius * 0.04, 0.01)
    # historical_size = 424
    # arcsecs = historical_size * pixscale

    # https://www.legacysurvey.org/viewer/jpeg-cutout?ra=190.1086&dec=1.2005&pixscale=0.27&bands=grz
    return f'https://www.legacysurvey.org/viewer/jpeg-cutout?ra={ra}&dec={dec}&pixscale={pixscale}&bands=grz&size={size}'


def crossmatch_coordinates(ra, dec, catalog_coords: SkyCoord):
    search_coord = SkyCoord(ra, dec, unit='deg')
    best_index, separation, _ = match_coordinates_sky(search_coord, catalog_coords)
    if separation > 10*u.arcsec:
        logging.warning('best index {} has large separation: {}'.format(best_index, separation))
    return best_index, separation


def find_neighbours(X, query_index, n_neighbors=36, metric='manhattan'):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', metric=metric).fit(X)
    _, indices = nbrs.kneighbors(X[query_index].reshape(1, -1))
    return np.squeeze(indices)  # ordered by similarity, will include itself


def show_galaxies(galaxies):
    st.write(galaxies)

    logging.info('Total galaxies: {}'.format(len(galaxies)))
    
    image_urls = list(galaxies.apply(get_url, axis=1))

    opening_html = '<div style=display:flex;flex-wrap:wrap>'
    closing_html = '</div>'
    child_html = ['<img src="{}" style=margin:3px;width:200px;></img>'.format(url) for url in image_urls]

    gallery_html = opening_html
    for child in child_html:
        gallery_html += child
    gallery_html += closing_html

    st.markdown(gallery_html, unsafe_allow_html=True)
    # st.markdown('<img src="{}"></img>'.format(child_html), unsafe_allow_html=True)
    # for image in images:
    #     st.image(image, width=250)


def show_query_galaxy(galaxy):

    image_col, data_col = st.beta_columns([2, 1])

    galaxy['url'] = get_url(galaxy)
    
    with image_col:
        st.header('Query Galaxy')
        st.image(galaxy['url'], width=200)
    
    with data_col:
        st.write('RA: {}'.format(galaxy['ra']))
        st.write('Dec: {}'.format(galaxy['dec']))
        st.write('Vizier: TODO')


def main(df, catalog_coords, features):

    st.title('Similarity Search')
    st.subheader('by Mike Walmsley ([@mike\_walmsley\_](https://twitter.com/mike_walmsley_))')

    ra = float(st.text_input('RA (deg)', key='ra', help='Right Ascension of galaxy to search (in degrees)', value='184.6750'))
    dec = float(st.text_input('Dec (deg)', key='dec', help='Declination of galaxy to search (in degrees)', value='11.7309'))

    best_index, separation = crossmatch_coordinates(ra, dec, catalog_coords)
    # TODO add a warning for big separation

    # TODO add metric option?
    neighbour_indices = find_neighbours(features, best_index)
    st.write(neighbour_indices)
    assert neighbour_indices[0] == best_index  # should find itself

    show_query_galaxy(df.iloc[best_index])

    show_galaxies(df.iloc[neighbour_indices[1:17]])


st.set_page_config(
    layout="wide",
    page_title='DECaLS Similarity',
    page_icon='gz_icon.jpeg'
)


if __name__ == '__main__':

    # logging.basicConfig(level=logging.CRITICAL)

    catalog = load_catalog()
    unaligned_features = load_features()
    # st.write('hello')
    # st.write(features.shape)
    # st.write(features[:10])

    # st.write(unaligned_features.head())
    df = pd.merge(catalog, unaligned_features, how='inner', on='png_loc')
    # if not len(df) == len(catalog):
    #     raise ValueError(len(df), len(catalog))
    feature_cols = [col for col in df.columns.values if col.endswith('_pca')]
    features = df[feature_cols].values


    catalog_coords = catalog_to_coordinates(df)


    main(df, catalog_coords, features)