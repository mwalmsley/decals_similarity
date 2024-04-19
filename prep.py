import pandas as pd
import numpy as np
from sklearn.decomposition import IncrementalPCA


if __name__ == '__main__':

    catalog_columns = ['brickid', 'objid', 'est_petro_th50']
    catalog = pd.read_parquet(
        '/Users/user/repos/download_DECaLS_images/catalogs/master_all_file_index_passes_file_checks.parquet',
        columns=catalog_columns
    )
    catalog['dr8_id'] = catalog['brickid'].astype(str) + '_' + catalog['objid'].astype(str)
    print(len(catalog))

    # # filter to more extended?

    # filter to featured?
    morph = pd.read_parquet(
        '/Users/user/Dropbox (The University of Manchester)/desi/catalogs/gz_desi_deep_learning_catalog_friendly.parquet', 
        columns=['brickid', 'objid', 'smooth-or-featured_smooth_fraction']
    )
    morph = morph[morph['smooth-or-featured_smooth_fraction'] < 0.75]
    morph['dr8_id'] = morph['brickid'].astype(str) + '_' + morph['objid'].astype(str)

    df = pd.merge(catalog, morph, on='dr8_id', how='inner', validate='1:1')
    del morph
    print(len(df))

    reps = pd.read_parquet('/Users/user/Downloads/representations_pca_40_with_coords.parquet')

    df = pd.merge(df, reps, on='dr8_id', how='inner', validate='1:1')
    print(len(df))
    del reps

    components = 20
    pca = IncrementalPCA(n_components=components, batch_size=10000)
    pca_cols = [col for col in df.columns.values if col.startswith('feat')]
    pca_results = pca.fit_transform(df[pca_cols])

    df = df[['dr8_id', 'est_petro_th50', 'ra', 'dec']]
    pca_results_cols = [f'feat_pca_{i}' for i in range(components)]
    df[pca_results_cols] = pca_results

    df = df.rename(columns={
        'est_petro_th50': 'estimated_radius',
        'dr8_id': 'galaxy_id'
    })
    df[pca_results_cols] = df[pca_results_cols].astype(np.float16)
    df.to_parquet(f'desi_pca{components}_with_radius_feat.parquet')
