import pytest

from pipeline_qc.camera_alignment import upload_aligned_files_to_fms as upload


# TODO: The `upload_aligned_files_to_fms.py` script warrants several more tests

@pytest.mark.parametrize("file_name", [
    # The actual values of the barcodes, scene, etc. are arbitrary. All that matters here is the date formatting
    '3500001000_100X_20200101-Scene-01-P01-A01.czi',
    '3500001000_100X-20200101-Scene-01-P01-A01.czi',
    '3500001000_100X-20200101_Scene-01-P01-A01.czi',
    '3500001000_100X_20200101_Scene-01-P01-A01.czi',
    '3520200101_100X_20200101_Scene-01-P01-A01.czi',
])
def test_get_zsd_and_date_happy(file_name):
    file = {
        'file': {
            'file_name': file_name,
            'original_path': r'\\\\allen\\aics\\microscopy\\PRODUCTION\\PIPELINE_4_4\\3500002666\\ZSD2\\100X_zstack\\Raw_Split_Scene\\3500002666_100X_20190121-Scene-56-P58-D10.czi'
        }
    }
    zsd, date = upload._grab_zsd_and_date(file)

    assert zsd == 'ZSD2'
    assert date == '20200101'


@pytest.mark.parametrize("file_name", [
    # The expectation here is that the file names result in some kind of exception
    '3500001000_100X~20200101~Scene-01-P01-A01.czi',
    '3500001000_100X_20200101Scene-01-P01-A01.czi'
])
def test_get_zsd_and_date_fail(file_name):
    with pytest.raises(Exception):
        file = {
            'file': {
                'file_name': file_name,
                'original_path': r'\\\\allen\\aics\\microscopy\\PRODUCTION\\PIPELINE_4_4\\3500002666\\ZSD2\\100X_zstack\\Raw_Split_Scene\\3500002666_100X_20190121-Scene-56-P58-D10.czi'
            }
        }
        upload._grab_zsd_and_date(file)
