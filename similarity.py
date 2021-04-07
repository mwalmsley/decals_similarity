import pickle
import logging
import base64
import os

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.units as u


def load_catalog():
    columns = ['ra', 'dec', 'estimated_radius', 'galaxy_id']

    # if LOCAL:
    #     catalog_loc = '/home/walml/repos/astronomaly/dr5_dr8_catalog_with_radius.parquet'
    # else:
    catalog_loc = 'dr5_dr8_catalog_with_radius.parquet'

    return pd.read_parquet(catalog_loc, columns=columns).reset_index(drop=True)
load_catalog = st.cache(load_catalog, persist=True, max_entries=1, allow_output_mutation=True, hash_funcs={pd.DataFrame: lambda _: None})


def load_features(num_components=10):
    columns = [f'feat_{n}_pca' for n in range(num_components)] + ['galaxy_id']

    # if LOCAL:
    #     features_loc = f'/home/walml/repos/astronomaly/dr5_8_b0_pca{num_components}_and_safe_ids.parquet'
    # else:
    features_loc = f'dr5_8_b0_pca{num_components}_and_safe_ids.parquet'

    return pd.read_parquet(features_loc, columns=columns)
load_features = st.cache(load_features, persist=True, max_entries=1, allow_output_mutation=True, hash_funcs={pd.DataFrame: lambda _: None})


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


def get_vizier_search_url(ra, dec):
    # http://simbad.u-strasbg.fr/simbad/sim-coo?Coord=180.4311d0.1828d&CooFrame=ICRS&CooEpoch=2000&CooEqui=2000&Radius=2&Radius.unit=arcmin&submit=submit+query&CoordList=
    return 'http://simbad.u-strasbg.fr/simbad/sim-coo?Coord={ra}d{dec}d&CooFrame=ICRS&CooEpoch=2000&CooEqui=2000&Radius=2&Radius.unit=arcmin&submit=submit+query&CoordList='


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

    logging.info('Total galaxies: {}'.format(len(galaxies)))
    
    galaxies['url'] = list(galaxies.apply(get_url, axis=1))

    show_galaxy_table(galaxies)
    st.text(" \n")

    opening_html = '<div style=display:flex;flex-wrap:wrap>'
    closing_html = '</div>'
    child_html = ['<img src="{}" style=margin:3px;width:200px;></img>'.format(url) for url in galaxies['url']]

    gallery_html = opening_html
    for child in child_html:
        gallery_html += child
    gallery_html += closing_html

    st.markdown(gallery_html, unsafe_allow_html=True)
    # st.markdown('<img src="{}"></img>'.format(child_html), unsafe_allow_html=True)
    # for image in images:
    #     st.image(image, width=250)


def show_galaxy_table(galaxies):

    # clean table
    galaxies = galaxies[['galaxy_id', 'ra', 'dec', 'url']].reset_index(drop=True)
    galaxies['link'] = galaxies['url'].apply(lambda x: make_clickable(x, text='Skyviewer Link'))

    display_table = galaxies[['galaxy_id', 'ra', 'dec', 'link']]
    display_table = display_table.rename(columns={'galaxy_id': 'Galaxy ID', 'ra': 'RA', 'dec': 'Dec', 'Link': 'link'})

    with st.beta_expander("Show table"):
        st.write(
            display_table.to_html(escape=False),
            unsafe_allow_html=True
        )

        table_to_download = galaxies.copy()   # TODO any extra tweaks, maybe more rows, etc
        csv = table_to_download.to_csv(index=False)
        # strings to b64 bytes
        b64 = base64.b64encode(csv.encode()).decode()

        st.markdown(
            f'<a href="data:file/csv;base64,{b64}" download="similar_galaxies.csv">Download table</a> of the 500 most similar galaxies.',
            unsafe_allow_html=True
        )
        st.markdown(r'Galaxy ID is either the IAUNAME (prefixed with J) for galaxies in DR5 and the NASA-Sloan Atlas v1_0_1, or formatted like {brickid}_{objid} otherwise')


def show_query_galaxy(galaxy):

    galaxy['url'] = get_url(galaxy)
    
    st.header('Closest Galaxy')

    # image_col, data_col = st.beta_columns([1, 1])

    # with image_col:
    st.image(galaxy['url'], width=200)
    
    # with data_col:
        # st.write(pd.DataFrame(data={'ra': 12, 'dec': 14, 'seperation': 1}))
        # st.write('RA: {:.5f}'.format(galaxy['ra']))
        # st.write('Dec: {:.5f}'.format(galaxy['dec']))
        # st.markdown())
    
    coords_string = 'RA: {:.5f}. Dec: {:.5f}'.format(galaxy['ra'], galaxy['dec'])
    viz_string = '[Search Vizier]({})'.format(get_vizier_search_url(galaxy['ra'], galaxy['dec']))
    
    st.write(coords_string + '. ' + viz_string)


# subfunctions are cached where possible
def prepare_data():

    catalog = load_catalog()
    unaligned_features = load_features()
    
    # merge is quite quick, no need to cache
    df = pd.merge(catalog, unaligned_features, how='inner', on='galaxy_id')

    feature_cols = [col for col in df.columns.values if col.endswith('_pca')]
    features = df[feature_cols].values

    catalog_coords = catalog_to_coordinates(df)

    return df, features, catalog_coords

# https://discuss.streamlit.io/t/display-urls-in-dataframe-column-as-a-clickable-hyperlink/743/8
def make_clickable(url, text):
    return f'<a target="_blank" href="{url}">{text}</a>'


def main():

    st.title('Similarity Search')
    st.subheader('by Mike Walmsley ([@mike\_walmsley\_](https://twitter.com/mike_walmsley_))')

    ra = float(st.text_input('RA (deg)', key='ra', help='Right Ascension of galaxy to search (in degrees)', value='184.6750'))
    dec = float(st.text_input('Dec (deg)', key='dec', help='Declination of galaxy to search (in degrees)', value='11.7309'))
    go = st.button('Search')

    # avoid doing a new search whenever ra OR dec changes, usually people would change both together
    if go:

        with st.spinner('Searching 911,442 galaxies. Please wait 30 seconds.'):
            
            # essentially all the delay
            # do this after rendering the inputs, so user has something to look at
            df, features, catalog_coords = prepare_data()

            best_index, separation = crossmatch_coordinates(ra, dec, catalog_coords)
        
            neighbour_indices = find_neighbours(features, best_index)
            assert neighbour_indices[0] == best_index  # should find itself

        show_query_galaxy(df.iloc[best_index])
        
        st.header('Similar Galaxies')

        show_galaxies(df.iloc[neighbour_indices[1:19]])


st.set_page_config(
    # layout="wide",
    page_title='DECaLS Similarity',
    page_icon='gz_icon.jpeg'
)


if __name__ == '__main__':

    # logging.basicConfig(level=logging.CRITICAL)

    LOCAL = os.getcwd() == '/home/walml/repos/decals_similarity'
    logging.info('Local: {}'.format(LOCAL))

    main()
