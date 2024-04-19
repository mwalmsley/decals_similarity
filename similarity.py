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
from huggingface_hub import hf_hub_download

# @st.cache_data(persist='disk', max_entries=1)
@st.cache_resource  # safe as never modified
def load_catalog(num_components=10):
    columns = ['galaxy_id', 'ra', 'dec', 'estimated_radius'] + [f'feat_pca_{n}' for n in range(num_components)]

    print('starting download')
    catalog_loc = hf_hub_download(
        repo_id='mwalmsley/zoobot-encoder-desi', 
        filename='desi_pca10_with_radius_feat.parquet', 
        repo_type="dataset"
    )

    print('started loading catalog')
    df = pd.read_parquet(catalog_loc, columns=columns).reset_index(drop=True)
    print('loaded within cache')
    return df


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
    return f'http://simbad.u-strasbg.fr/simbad/sim-coo?Coord={ra}d{dec}d&CooFrame=ICRS&CooEpoch=2000&CooEqui=2000&Radius=2&Radius.unit=arcmin&submit=submit+query&CoordList='


def separation_warning(separation):
    if separation > 1:
        st.warning('Warning - best matching galaxy has extremely large separation: {} deg. Are your coordinates in the DESI-LS footprint below?'.format(separation))
        st.image('sky_coverage.png')
    elif separation > (20/3600):
        st.warning('Warning - best matching galaxy has moderately large separation: {} arcmin. The target galaxy may not be in our DESI-LS catalog. Is it between r-mag 14.0 and 19?'.format(separation/3600))


def find_neighbours_from_index(X, query_index, n_neighbors=16, metric='manhattan'):
    # n_neighbours only controls top k, no smoothing remember
    print('fitting index tree')
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', metric=metric).fit(X)
    _, indices = nbrs.kneighbors(X[query_index].reshape(1, -1))
    return np.squeeze(indices)  # ordered by similarity, will include itself

def find_neighbours_from_query(X, query, n_neighbors=1, metric='euclidean'):
    print('fitting query tree')
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', metric=metric).fit(X)
    distances, indices = nbrs.kneighbors(query)
    return np.squeeze(distances), np.squeeze(indices)  # ordered by similarity, will include itself


def show_galaxies(galaxies, max_display_galaxies=18):
    
    galaxies['url'] = list(galaxies.apply(get_url, axis=1))

    show_galaxy_table(galaxies, max_display_galaxies)
    st.text(" \n")

    opening_html = '<div style=display:flex;flex-wrap:wrap>'
    closing_html = '</div>'
    child_html = ['<img src="{}" style=margin:3px;width:200px;></img>'.format(url) for url in galaxies['url'][:max_display_galaxies]]

    gallery_html = opening_html
    for child in child_html:
        gallery_html += child
    gallery_html += closing_html

    st.markdown(gallery_html, unsafe_allow_html=True)
    # st.markdown('<img src="{}"></img>'.format(child_html), unsafe_allow_html=True)
    # for image in images:
    #     st.image(image, width=250)


def show_galaxy_table(galaxies, max_display_galaxies):

    # clean table
    galaxies = galaxies[['galaxy_id', 'ra', 'dec', 'url']].reset_index(drop=True)
    galaxies['link'] = galaxies['url'].apply(lambda x: make_clickable(x, text='Skyviewer Link'))

    display_table = galaxies[['galaxy_id', 'ra', 'dec', 'link']][:max_display_galaxies]
    display_table = display_table.rename(columns={'galaxy_id': 'Galaxy ID', 'ra': 'RA', 'dec': 'Dec', 'link': 'Link'})

    with st.expander("Show table"):
        st.write(
            display_table.to_html(escape=False),
            unsafe_allow_html=True
        )

        table_to_download = galaxies.copy()   # TODO any extra tweaks, maybe more rows, etc
        csv = table_to_download.to_csv(index=False)
        # strings to b64 bytes
        b64 = base64.b64encode(csv.encode()).decode()

        # TODO replace with this
        st.markdown(
            f'<a href="data:file/csv;base64,{b64}" download="similar_galaxies.csv">Download table</a> of the 500 most similar galaxies.',
            unsafe_allow_html=True
        )
        st.markdown(r'Galaxy ID is formatted like {brickid}\_{objid} in DESI-LS DR8. Crossmatch to DESI-LS DR8 with 8000\_{brickid}\_{objid}')


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

    df = load_catalog(num_components=10)

    feature_cols = [col for col in df.columns.values if col.startswith('feat_pca')]
    features = df[feature_cols].values

    return df, features

# https://discuss.streamlit.io/t/display-urls-in-dataframe-column-as-a-clickable-hyperlink/743/8
def make_clickable(url, text):
    return f'<a target="_blank" href="{url}">{text}</a>'


def get_nontrivial_neighbours(query_galaxy: pd.Series, neighbours: pd.DataFrame, min_sep=100*u.arcsec):
    query_coord = SkyCoord(query_galaxy['ra'], query_galaxy['dec'], unit='deg')
    separations = [SkyCoord(other['ra'], other['dec'], unit='deg').separation(query_coord) for _, other in neighbours.iterrows()]
    above_min_sep = np.array([sep > min_sep for sep in separations])
    if np.any(~above_min_sep):
        st.warning(
            """
            Removed {} sources within 100 arcsec of target galaxy.
            These are likely to be trivial matches from nearby catalog entries with similar field-of-view.
            """.format((~above_min_sep).sum())
        )
    return neighbours[above_min_sep].reset_index(drop=True)


def main():

    st.title('Similarity Search')
    st.subheader('by Mike Walmsley ([@mike\_walmsley\_](https://twitter.com/mike_walmsley_))')
    st.text(" \n")

    ra = float(st.text_input('RA (deg)', key='ra', help='Right Ascension of galaxy to search (in degrees)', value='184.6750'))
    dec = float(st.text_input('Dec (deg)', key='dec', help='Declination of galaxy to search (in degrees)', value='11.73181'))

    legacysurvey_str_default = 'https://www.legacysurvey.org/viewer?ra=184.6750&dec=11.73181&layer=ls-dr8&zoom=12'
    legacysurvey_str = st.text_input('or, paste link', key='legacysurvey_str', help='Legacy Survey Viewer link (for ra and dec)', value=legacysurvey_str_default)
    
    if legacysurvey_str != legacysurvey_str_default:
        query_params = legacysurvey_str.split('?')[-1]
        ra_str = query_params.split('&')[0].replace('ra=', '')
        dec_str = query_params.split('&')[1].replace('dec=', '')
        ra = float(ra_str)
        dec = float(dec_str)
        st.markdown(f'Using link coordinates: {ra}, {dec}')

    with st.spinner('Loading representation, please wait'):
        # essentially all the delay
        # do this after rendering the inputs, so user has something to look at
        df, features = prepare_data()
        print('data ready')
        go = st.button('Search')
        # st.markdown('Ready to search.')

    with st.expander('Important Notes'):
        st.markdown(
            """
            Which galaxies are included?
            - Galaxies must be between r-mag 14.0 and 19 (the SDSS spectroscopic limit).
            - Galaxies must be extended enough to be included in Galaxy Zoo (roughly, petrosian radius > 3 arcseconds)
            - Galaxies must be in the DECaLS DR8 sky area. A sky area chart will display if the target coordinates are far outside.
            
            What are the machine learning limitations?
            - The underlying model does not receive colour information to avoid bias. Colour grz images are shown for illustration only.
            - The underlying model is likely to perform better with "macro" morphology (e.g. disturbances, rings, etc.) than small anomalies in otherwise normal galaxies (e.g. supernovae, Voorwerpen, etc.)
            - Finding no similar galaxies does not imply there are none.
            
            Please see the paper (in prep.) for more details.
            """
        )
    st.text(" \n")



    # avoid doing a new search whenever ra OR dec changes, usually people would change both together
    if go:

        with st.spinner(f'Searching {len(df)} galaxies.'):
            
            coordinate_query = np.array([ra, dec]).reshape((1, -1))
            separation, best_index = find_neighbours_from_query(df[['ra', 'dec']], coordinate_query)  # n_neigbours=1
            # print('crossmatched')

            separation_warning(separation)
        
            neighbour_indices = find_neighbours_from_index(features, best_index)
            assert neighbour_indices[0] == best_index  # should find itself

            query_galaxy = df.iloc[best_index]
            neighbours = df.iloc[neighbour_indices[1:]]

            # exclude galaxies very very close to the original
            # sometimes catalog will record one extended galaxy as multiple sources
            nontrivial_neighbours = get_nontrivial_neighbours(query_galaxy, neighbours)

        show_query_galaxy(query_galaxy)
        
        st.header('Similar Galaxies')

        show_galaxies(nontrivial_neighbours, max_display_galaxies=18)


st.set_page_config(
    # layout="wide",
    page_title='DECaLS Similarity',
    page_icon='gz_icon.jpeg'
)



if __name__ == '__main__':

    # logging.basicConfig(level=logging.CRITICAL)


    # streamlit run similarity.py --server.fileWatcherType none


    # LOCAL = os.getcwd() == '/home/walml/repos/decals_similarity'
    # logging.info('Local: {}'.format(LOCAL))

    main()

