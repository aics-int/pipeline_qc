#!/usr/bin/env python

from collections import namedtuple
import pathlib

from lkaccess.accessors import FOV
import numpy.testing
import pytest
import skimage.measure

import pipeline_qc.cell_generation.labkey_cell_generation as CG


##################################################
# Constants
##################################################

FOV_ID = 12
FOV_WIDTH = 924.0
FOV_HEIGHT = 624.0
ALGORITHM_ID = 13
PIXEL_UNIT_ID = 8
ORIGIN_ID = 34
RUN_ID = 9001
NUC_FILE_ID = 'nuc_file_id'
MEMB_FILE_ID = 'memb_file_id'
ImageData = namedtuple('ImageData', ['nuc', 'memb'])
RESOURCE_FOLDER = pathlib.Path(__file__).absolute().parent / 'resources'


##################################################
# Fixtures
##################################################

@pytest.fixture(scope='module')
def membrane_image_path():
    return str(RESOURCE_FOLDER / '3500001234_100X_20170825_1-Scene-20-P21-E06.ome_segmentation.tiff')


@pytest.fixture(scope='module')
def image_data(membrane_image_path):
    return CG._get_image_data(membrane_image_path)


@pytest.fixture(scope='module')
def image_data_no_cells(membrane_image_path):
    image_data = CG._get_image_data(membrane_image_path)
    image_data[:, :, :] = 0
    return image_data


@pytest.fixture(scope='module')
def regionprops(image_data: ImageData):
    return skimage.measure.regionprops(image_data)


@pytest.fixture
def generic_metadata():
    # These values __should__ exist in staging
    return {
        "file": {
            "file_id": "random_file_id_string_value"
        },
        "microscopy": {
            "fov_id": FOV_ID
        },
        "content_processing": {
            "channels": {
                "0": {
                    "algorithm_id": ALGORITHM_ID,
                    "run_id": RUN_ID
                },
                "1": {
                    "algorithm_id": ALGORITHM_ID,
                    "run_id": RUN_ID
                }
            }
        }
    }


@pytest.fixture
def membrane_metadata(generic_metadata, membrane_image_path):
    memb = generic_metadata.copy()
    memb['file']['file_name'] = str(pathlib.Path(membrane_image_path).name)
    return memb


@pytest.fixture(scope='module')
def fov() -> FOV:
    return FOV.from_labkey({
        'OriginXY': 1,
        'ROIId': None,
        'Created': '2018/05/05 01:10:16',
        'DimensionZ': 65.0,
        'DimensionX': FOV_WIDTH,
        'FOVId': 12,
        'DimensionY': FOV_HEIGHT,
        'OriginZ': 1,
        'CenterZ': 11228.395,
        'CenterY': 41031.794,
        'Objective': 100.0,
        'CenterX': 53868.022,
        'DimensionUnits': 8,
        'CreatedBy': 1051,
        'InstrumentId': 5,
        'PixelScaleZ': 0.29,
        'PixelScaleUnits': 3,
        'PixelScaleX': 0.10833333333333332,
        'PixelScaleY': 0.10833333333333332,
        'CenterUnits': 3,
        'QCStatusId': 6,
        'FOVImageDate': '2017/06/23 17:31:12',
        'SourceImageFileId': '761c2aff82d14737b2813cfe0c7ceb25',
        'WellId': 24822
    })


##################################################
# Tests
##################################################
def test_check_metadata_blocks(membrane_metadata):
    # Shouldn't raise an error
    CG._check_metadata_blocks(membrane_metadata)


def test_check_metadata_blocks_empty():
    with pytest.raises(AssertionError):
        CG._check_metadata_blocks({})


def test_check_metadata_blocks_missing_processing():
    with pytest.raises(AssertionError):
        CG._check_metadata_blocks({'microscopy': {}})


def test_check_metadata_blocks_missing_microscopy():
    with pytest.raises(AssertionError):
        CG._check_metadata_blocks({'content_processing': {}})


def test_check_metadata_blocks_no_fov(membrane_metadata):
    with pytest.raises(AssertionError):
        memb = membrane_metadata.copy()
        memb['microscopy'].pop('fov_id')
        CG._check_metadata_blocks(membrane_metadata)


def test_check_metadata_blocks_no_channels(membrane_metadata):
    with pytest.raises(AssertionError):
        memb = membrane_metadata.copy()
        memb['content_processing'].pop('channels')
        CG._check_metadata_blocks(membrane_metadata)


def test_check_metadata_blocks_no_channel_zero(membrane_metadata):
    with pytest.raises(AssertionError):
        memb = membrane_metadata.copy()
        memb['content_processing']['channels'].pop('0')
        CG._check_metadata_blocks(membrane_metadata)


def test_check_metadata_blocks_no_algorithm(membrane_metadata):
    with pytest.raises(AssertionError):
        memb = membrane_metadata.copy()
        memb['content_processing']['channels']['0'].pop('algorithm_id')
        CG._check_metadata_blocks(membrane_metadata)


def test_get_image_data(membrane_image_path, image_data):
    for expected, actual in zip(image_data,
                                CG._get_image_data(membrane_image_path)):
        numpy.testing.assert_array_equal(expected, actual)


def test_region_props(image_data, regionprops):
    actual_props = CG._get_region_properties(image_data)
    for expected, actual in zip(regionprops, actual_props):
        assert expected.bbox == actual.bbox
        assert expected.label == actual.label
        assert expected.image.shape == actual.image.shape


def test_make_cells(regionprops, fov):
    cells = CG._make_cells(FOV_ID, regionprops, PIXEL_UNIT_ID, ORIGIN_ID,
                           fov, ALGORITHM_ID, RUN_ID, MEMB_FILE_ID, FOV_WIDTH, FOV_HEIGHT)
    labels = numpy.array([c.cell_index for c in cells])
    # Make sure all labels are unique
    numpy.testing.assert_array_equal(labels, numpy.unique(labels))

    for region, cell in zip(regionprops, cells):
        assert cell.cell_index == region.label
        assert cell.center_x == region.bbox[2]
        assert cell.center_y == region.bbox[1]
        assert cell.center_z == region.bbox[0]
        assert cell.dimension_x == region.image.shape[2]
        assert cell.dimension_y == region.image.shape[1]
        assert cell.dimension_z == region.image.shape[0]
        assert cell.fov_id == FOV_ID
        assert cell.center_units == cell.dimension_units == PIXEL_UNIT_ID
        assert cell.content_generation_algorithm_id == ALGORITHM_ID
        assert cell.origin_xy == cell.origin_z == ORIGIN_ID
        assert cell.source_membrane_file_id == MEMB_FILE_ID


def test_cells_fov_edge(image_data, fov):
    region_props = CG._get_region_properties(image_data)
    cells = CG._make_cells(FOV_ID, region_props, PIXEL_UNIT_ID, ORIGIN_ID, fov, ALGORITHM_ID,
                           RUN_ID, MEMB_FILE_ID, FOV_WIDTH, FOV_HEIGHT)
    cells_with_fovedge_false = [cell for cell in cells if cell.fov_edge is False]
    assert len(cells) == 17  # These values are somewhat arbitrary, only based on current resource file being used
    assert len(cells_with_fovedge_false) == 3


def test_no_cells(image_data_no_cells, fov):
    region_props = CG._get_region_properties(image_data_no_cells)
    cells = CG._make_cells(FOV_ID, region_props, PIXEL_UNIT_ID, ORIGIN_ID, fov, ALGORITHM_ID,
                           RUN_ID, MEMB_FILE_ID, FOV_WIDTH, FOV_HEIGHT)

    # This cell map should now be empty, and this call shouldn't throw an error
    cell_map = CG._get_or_create_cells(cells)
    assert cell_map == {}
