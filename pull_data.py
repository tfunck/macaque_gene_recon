import subprocess
import os
import json
import numpy as np
import pandas as pd
import nibabel as nib
import ants
import matplotlib.pyplot as plt
from skimage.draw import polygon





def pull_data(section_n_list, output_dir) -> list:
    
    os.makedirs(output_dir, exist_ok=True)  

    file_list = []
    for i in section_n_list :

        filename = f'brain_region_T{i}_macaque_f001_Mq179-20230428.json'
        local_fin = f'{output_dir}/{filename}'
        web_add = f'https://macaque.digital-brain.cn/cell/{filename}'

        if not os.path.exists(local_fin):
            subprocess.run(f'wget {web_add} -O {local_fin}', shell=True, executable='/bin/bash' )

        if not os.path.exists(local_fin):
            print(f'Error: {filename} not found')
            local_fin = None

        file_list.append(local_fin)

    return file_list

def get_region_id(file_list:list) :
    region_names = []
    for file in file_list:
        data = json.load(open(file,'r'))
        for region in data['regions']:
            region_names.append(region['region_id'])
    
    return region_names

def plot_region(region:dict, array_all:np.array, id:int, xdim:int, ydim:int, minx:int, miny:int):

    array = np.zeros((xdim, ydim))
    x = np.rint( (np.array(region['rx']) - minx)/downsample_factor ).astype(int)
    y = np.rint( (np.array(region['ry']) - miny)/downsample_factor ).astype(int)

    assert np.min(x) >= 0, f'x min {np.min(x)}'
    assert np.min(y) >= 0, f'y min {np.min(y)}'
    assert np.max(x) < xdim, f'x max {np.max(x)}'
    assert np.max(y) < ydim, f'y max {np.max(y)}'

    array[x, y] = id

    # fill polygon defined by points

    rr, cc = polygon(x, y, array.shape)

    array_all[rr, cc] = id

    return array_all

def plot_section(local_fin:str, output_dir:str, clobber:bool=False) -> str:

    os.makedirs(output_dir, exist_ok=True)

    output_fin = f'{output_dir}/{local_fin.split("/")[-1].replace(".json", ".nii.gz")}'

    if not os.path.exists(output_fin) or clobber :
        data = json.load(open(local_fin,'r'))

        regions = data['regions']

        maxx = np.max([ np.max(region['rx']) for region in regions ]) 
        maxy = np.max([ np.max(region['ry']) for region in regions ])
        minx = np.min([ np.min(region['rx']) for region in regions ])
        miny = np.min([ np.min(region['ry']) for region in regions ]) 

        xdim = np.ceil((maxx-minx)/downsample_factor).astype(int) + 2
        ydim = np.ceil((maxy-miny)/downsample_factor).astype(int) + 2

        array_all = np.zeros((xdim, ydim))

        for _, region in enumerate(regions):
            id = region['region_id']
            array_all  = plot_region(region, array_all, id, xdim, ydim, minx, miny)

        array_all = np.flip(array_all, axis=1)

        nib.Nifti1Image(array_all, np.eye(4)).to_filename(output_fin)
    
    return output_fin


def pad_sections(df:pd.DataFrame, output_dir:str, border:float=1.2) -> pd.DataFrame:

    os.makedirs(output_dir, exist_ok=True)

    xdim = 0
    zdim = 0

    for section_fin in df['section']:
        section = nib.load(section_fin).get_fdata()
        xdim = max(xdim, section.shape[0])
        zdim = max(zdim, section.shape[1])

    # add border to each section
    xdim = int(xdim*border)
    zdim = int(zdim*border)
    
    indices = range(df.shape[0])

    for i in indices:
        row = df.iloc[i]

        section = nib.load(row['section']).get_fdata()
        
        if row['flip']:
            section = np.flip(section, axis=1)

        section_padded = np.zeros((xdim, zdim))

        # pad section to match volume dimensions and center it
        xpad = xdim - section.shape[0]
        zpad = zdim - section.shape[1]
        xstart = xpad // 2
        zstart = zpad // 2

        section_padded[xstart:xstart+section.shape[0], zstart:zstart+section.shape[1]] = section

        section_padded_fin = f'{output_dir}/{os.path.basename(row["section"])}'

        nib.Nifti1Image(section_padded, np.eye(4)).to_filename(section_padded_fin)

        df['section'].iloc[i] = section_padded_fin
        print(section_padded_fin)
    
    
    return df
    
def sections_to_volume(df:pd.DataFrame, affine:np.array) :
    """Load sections in df and create a volume. The volume has xdim, zdim from the max and min values of the sections. 
    The ydim is max value of 'n' in the df. The volume is saved as a nifti volume."""

    # get max and min values of x and y

    volume_fin = 'volume.nii.gz'
    volume_interp_fin = 'volume_interp.nii.gz'

    volume = None

    
    assert df['y'].nunique() == df.shape[0], 'Error: y values are not unique'

    if not os.path.exists(volume_fin) :
        xdim , zdim = nib.load(df['section'].iloc[0]).shape

        print(df['y_mm'])

        ydim = df['y'].max() + 1

        volume = np.zeros((xdim, ydim, zdim))

        for _, row in df.iterrows():
            y = row['y']
            section_fin = row['section']
            print('y', y, section_fin)
            section = nib.load(section_fin).get_fdata()
            
            if row['flip']:
                section = np.flip(section, axis=1)

            # pad section to match volume dimensions and center it
            xpad = xdim - section.shape[0]
            zpad = zdim - section.shape[1]
            xstart = xpad // 2
            zstart = zpad // 2
            volume[xstart:xstart+section.shape[0],y,zstart:zstart+section.shape[1]] = section


        volume = np.flip(volume, axis=2)

        nib.Nifti1Image(volume, affine).to_filename(volume_fin)

    if not os.path.exists(volume_interp_fin) :
        if volume is None:
            img = nib.load(volume_fin)
            volume = img.get_fdata()
            affine = img.affine
        
        volume_interp = interpolate_missing_sections(volume)

        nib.Nifti1Image(volume_interp, affine).to_filename(volume_interp_fin)

    return section_df

def align_sections(df:pd.DataFrame, output_dir:str):
    """Align sections in df. The alignment is done using ANTs."""

    os.makedirs(output_dir, exist_ok=True)

    indices = range(df.shape[0])

    for i in indices:
        
        j = i + 1
        
        if j >= df.shape[0]:
            break

        fixed_image = df['section'].iloc[i]    
        moving_image = df['section'].iloc[j]

        output_base = f'{output_dir}/{os.path.basename(moving_image).replace(".nii.gz", "_aligned")}'
        output_fin = f'{output_base}.nii.gz'
        qc_fin = output_base+'.png'
        #cmd = f'antsRegistration -d 2 -v 1 -r [{fixed_image},{moving_image},1] -m GC [{fixed_image},{moving_image},1,4] -t Rigid[0.1] -c 1000x500x250x100 -f 4x2x1 -s 2x1x0 -u 1 -z 1 -o [output, {output_fin}]', 
        #print(cmd)

        #subprocess.run(
        #        cmd,
        #        shell=True, 
        #        executable='/bin/bash'
        #    )
        if not os.path.exists(output_fin) or not os.path.exists(qc_fin) :        
            reg = ants.registration(
                fixed=ants.image_read(fixed_image), 
                moving=ants.image_read(moving_image), 
                output=output_base, 
                aff_metric='GC',  
                aff_iterations='1000x500x250x100',
                aff_shrink_factors='5x4x3x2x1',
                aff_smoothing_sigmas='2.5x2x1.5x1x0',
                type_of_transform='Rigid', 
                verbose=True
                )
        
            out_vol = reg['warpedmovout'].numpy()

            nib.Nifti1Image(out_vol, nib.load(moving_image).affine).to_filename(output_fin)

            plt.imshow(out_vol)
            plt.imshow(nib.load(fixed_image).get_fdata(), alpha=0.5)
            plt.savefig(qc_fin)

        df['section'].iloc[j] = output_fin
    return df


def interpolate_missing_sections(
    vol: np.array, 
) -> np.array:
    """Interpolates missing sections in a volume.

    :param vol (ndarray): The input volume.
    :dilate_volume (bool, optional): Whether to dilate the volume before interpolation. Defaults to False.
    :return ndarray: The volume with missing sections interpolated.
    """
    from brainbuilder.utils.utils import get_section_intervals

    intervals = get_section_intervals(vol)

    out_vol = vol.copy()
    for i in range(len(intervals) - 1):
        j = i + 1

        x0, x1 = intervals[i]  # intervals of consecutive sections
        y0, y1 = intervals[j]  # intervals of consecutive sections
        x1 -= 1

        x = vol[:,x1,:] 

        y = vol[:,y0,:]

        for ii in range(x1+1, y0):
            den = y0 - x1
            assert den != 0, "Error: 0 denominator when interpolating missing sections"
            
            d = float(ii - x1) / den
            # Linear interpolation
            z = x * (1 - d) + d * y
            # Nearest neighbor interpolation
            #if d < 0.5:
            #    z = x
            #else:
            #    z = y
            
            print(x1, d, ii, y0, '-->', np.mean(x), np.mean(z), np.mean(y))

            out_vol[:, ii, :] = z

    return out_vol

def get_affine_dimensions(ymax,ymin,xdim,ydim,zdim): 
    affine = np.eye(4)

    yrange = ymax - ymin

    template_yrange = 47+30
    template_xrange = 60
    template_zrange = 47

    xy_ratio = template_xrange / template_yrange
    zy_ratio = template_zrange / template_yrange

    xrange = yrange * xy_ratio
    zrange = yrange * zy_ratio

    xres = xrange / xdim 
    yres = yrange / ydim
    zres = zrange / zdim 

    affine[0,0] = xres
    affine[1,1] = yres
    affine[2,2] = zres

    return affine
        

if __name__ == '__main__':

    downsample_factor = 400

    section_list_csv = 'gene_section_list.csv'
    clobber = True

    output_dir = 'outputs'

    section_df = pd.read_csv(section_list_csv)

    yres = 0.02 #(section_df['y'].max() - section_df['y'].min()) / downsample_factor

    section_json_fin = pull_data(section_df['section'].values, output_dir+'/json/')

    section_df['json'] = section_json_fin
    
    # read json file with:
    #  'regions' key and 'x' and 'y' sub-keys
    #   sec_para" key and "maxx", "maxy", "minx", "miny" sub-keys

    section_df['flip'] = False

    section_fin_list = []
    for local_fin in section_json_fin:
        
        section_fin = plot_section(local_fin, output_dir+'/sections/', clobber=clobber)

        section_fin_list.append(section_fin)

    section_df['section'] = section_fin_list


    x0 = np.min(np.abs(np.diff(section_df['y_mm'].values)))
    x1 = np.min(np.abs(np.diff(section_df['y_mm'].values[::-1])))

    #yres = min(x0, x1)

    section_df['y'] = np.rint( ( section_df['y_mm'] - section_df['y_mm'].min() ) / yres  ).astype(int)

    # drop rows with repeat y values
    section_df = section_df.drop_duplicates(subset='y')

    # reverse y values
    section_df['y'] = section_df['y'].max() - section_df['y']

    if section_df['y'].nunique() != section_df.shape[0]:
        section_df = section_df.drop_duplicates(subset='y')

    xdim = nib.load(section_df['section'].iloc[0]).shape[0]
    ydim = section_df['y'].max() + 1
    zdim = nib.load(section_df['section'].iloc[0]).shape[1]

    affine = get_affine_dimensions(section_df['y'].max(), section_df['y'].min(), xdim,ydim,zdim)


    assert section_df['y'].nunique() == section_df.shape[0], 'Error: y values are not unique'

    section_df = pad_sections(section_df.copy(), output_dir+'/padded/')

    #section_df = align_sections(section_df, output_dir+'/aligned/')
    section_df = sections_to_volume(section_df)
    
    # Create Hemisphere DataFrame with sub, hemisphere, struct_ref_vol, gm_surf, wm_surf
    hemi_df = pd.DataFrame({
        'sub': ['f001'],
        'hemisphere': ['L'],
        'struct_ref_vol': ['data/mebrains/MEBRAINS_segmentation_NEW_gm_left.nii.gz'],
        'gm_surf': ['data/yerkes19/lh.MEBRAINS_0.5mm_0.0.surf.gii'],
        'wm_surf': ['data/yerkes19/lh.MEBRAINS_0.5mm_1.0.surf.gii']
    })
    hemi_df.to_csv('data/hemisphere.csv', index=False)

    # Create Chunk DataFrame with sub, hemisphere, chunk, pixel_size_0, pixel_size_1, section_thickness, direction
    chunk_df = pd.DataFrame({
        'sub': ['f001'],
        'hemisphere': ['L'],
        'chunk': [1],
        'pixel_size_0': [0.7],
        'pixel_size_1': [0.7],
        'section_thickness': [0.02],
        'direction': ['caudal_to_rostral']
    }) 
    chunk_df.to_csv('data/chunk.csv', index=False)

    # Create Section DataFrame with sub, hemisphere, chunk, aquisition, sample, and raw
    section_df['sub'] = 'f001'
    section_df['hemisphere'] = 'L'
    section_df['chunk'] = 1
    section_df['acquisition'] = 'synth'
    section_df['sample'] = section_df['y']
    section_df['raw'] = section_df['section']
    section_df.to_csv('data/section.csv', index=False)

