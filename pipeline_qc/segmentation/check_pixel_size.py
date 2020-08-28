import aicsimageio
from pipeline_qc.image_qc_methods.query_fovs import query_fovs
from pipeline_qc.segmentation.common.fov_file import FovFile

def main():
    #aicsimageio.use_dask(False)

    df = query_fovs(workflows=["Pipeline 4", "Pipeline 4.1", "Pipeline 4.2", "Pipeline 4.4"])
    sizes = []

    for index, row in df.iterrows():
        fov = FovFile.from_dataframe_row(row)
        print(f"Reading {fov.local_file_path}")
        img = aicsimageio.AICSImage(fov.local_file_path)
        size = str(img.get_physical_pixel_size())   
        print(size)
        if size not in sizes:
            sizes.append(size)
            with open("output.txt", "a") as output:
                output.write(f"{size}\n")

    print(sizes)

if __name__ == "__main__":
    main()