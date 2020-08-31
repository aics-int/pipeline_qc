#!/usr/bin/env python
"""
Input:
 - One segmentation file with the following channels:
 -  [0] - Nucleus segmentation
 -  [1] - Membrane segmentation
 -  [2] - Nucleus contour
 -  [3] - Membrane contour

"""
import json
import logging
import typing

from aicsimageio import AICSImage
from aicsfiles import FileManagementSystem
from datetime import datetime
from lkaccess import LabKey, QueryFilter
from lkaccess.accessors import FOV, Cell
from pandas import DataFrame
import skimage.measure

ORIGIN = 'Pixel coordinate from zero index'
PIXEL_UNIT = 'pixels'
MICROSCOPY = 'microscopy'
PROCESSING = 'processing'
CELL = 'Cell'

NUCLEUS_SEGMENTATION_CHANNEL_INDEX = 0
MEMBRANE_SEGMENTATION_CHANNEL_INDEX = 1
NUCLEUS_CONTOUR_CHANNEL_INDEX = 2
MEMBRANE_CONTOUR_CHANNEL_INDEX = 3

'''
 The number of pixels a given cell may be away from the FOV image boundary yet still be classified as
 touching the edge of the FOV.
 In other words, cells with bounding boxes within CELL_BOUNDARY_PIXEl_PRECISION_OFFSET of the FOV edge will have
 'fov_edge' set to 'True'
'''
CELL_BOUNDARY_PIXEL_PRECISION_OFFSET = 4

log = logging.getLogger(__name__)
logging.getLogger(__name__).setLevel(logging.INFO)

CellList = typing.List[Cell]


def _check_metadata_blocks(segmentation_file_metadata):
    # Ensure the metadata argument contains all necessary fields
    # Subject to change!
    assert "file" in segmentation_file_metadata, "['file'] block missing from file metadata"
    assert "file_id" in segmentation_file_metadata['file'], "['file_id'] missing from metadata ['file']"

    assert "microscopy" in segmentation_file_metadata, "['microscopy'] block missing from file metadata"
    assert "fov_id" in segmentation_file_metadata['microscopy'], "['fov_id'] block missing from metadata ['microscopy']"

    assert "content_processing" in segmentation_file_metadata, "['content_processing'] missing from file metadata"
    assert "channels" in segmentation_file_metadata['content_processing'], \
        "'channels' block missing from metadata ['content_processing']"
    assert "0" in segmentation_file_metadata['content_processing']['channels'], \
        "Channel 0 missing from file metadata ['content_processing']['channels']"
    assert "1" in segmentation_file_metadata['content_processing']['channels'], \
        "Channel 1 missing from file metadata ['content_processing']['channels']"

    channels = segmentation_file_metadata['content_processing']['channels']
    for channel in channels:
        assert 'algorithm_id' in channels[channel], f"['algorithm_id'[ missing from channel {channel}"
        assert 'run_id' in channels[channel], f"['run_id'] missing from channel {channel}"


def _origin_id_from_name(name, lk_conn: LabKey):
    log.debug("Attempting to retrieve Origin with name '{}'".format(name))
    response = lk_conn.select_first(schema_name=MICROSCOPY,
                                    query_name='Origin',
                                    filter_array=[
                                        QueryFilter('Name', name, 'eq')
                                    ])

    return response['OriginId']


def _unit_id_from_name(name, lk_conn: LabKey):
    log.debug("Attempting to retrieve Unit with name '{}'".format(name))
    response = lk_conn.select_first(schema_name=MICROSCOPY,
                                    query_name='Units',
                                    filter_array=[
                                        QueryFilter('Name', name, 'eq')
                                    ])

    return response['UnitsId']


def _get_image_data(segmentation_file_path):
    # Generate AICSImage object for membrane segmentation
    # Membrane segmentation data is expected to be on channel 1
    memb_image = AICSImage(segmentation_file_path)
    memb_seg_data = memb_image.get_image_data("ZYX", T=0, C=MEMBRANE_SEGMENTATION_CHANNEL_INDEX)

    return memb_seg_data


def _get_region_properties(memb_data):
    return skimage.measure.regionprops(memb_data)


def _make_cell(fov_id, region, pixel_unit_id, origin_id, fov: FOV, algorithm_id,
               run_id, segmentation_file_id, fov_width, fov_height):
    cell = Cell(fov_id=fov_id,
                center_units=pixel_unit_id,
                origin_xy=origin_id,
                origin_z=origin_id,
                pixel_scale_x=fov.pixel_scale_x,
                pixel_scale_y=fov.pixel_scale_y,
                pixel_scale_z=fov.pixel_scale_z,
                pixel_scale_units=pixel_unit_id,
                dimension_units=pixel_unit_id,
                content_generation_algorithm_id=algorithm_id,
                run_id=run_id,
                source_nucleus_file_id=segmentation_file_id,
                source_membrane_file_id=segmentation_file_id,
                nucleus_segmentation_channel_index=NUCLEUS_SEGMENTATION_CHANNEL_INDEX,
                membrane_segmentation_channel_index=MEMBRANE_SEGMENTATION_CHANNEL_INDEX,
                nucleus_contour_channel_index=NUCLEUS_CONTOUR_CHANNEL_INDEX,
                membrane_contour_channel_index=MEMBRANE_CONTOUR_CHANNEL_INDEX
                )

    # The cell index is defined as the label of the property region
    cell.cell_index = region.label

    # bbox gives us 6 values, the start ZYX closest to [0, 0, 0],
    # and the end ZYX furthest from the origin. We use the first
    # 3 as the origin for the cell
    bbox_origin = region.bbox[:3]
    cell.center_z, cell.center_y, cell.center_x = bbox_origin

    # The shape of the bounding box will give us the XYZ dimensions
    bbox_dims = region.image.shape
    cell.dimension_z, cell.dimension_y, cell.dimension_x = bbox_dims

    # If a cell's bounding box touches or is near any edge of the FOV, mark FovEdge as "True"
    if cell.center_x <= CELL_BOUNDARY_PIXEL_PRECISION_OFFSET \
            or cell.center_y <= CELL_BOUNDARY_PIXEL_PRECISION_OFFSET \
            or cell.center_z <= CELL_BOUNDARY_PIXEL_PRECISION_OFFSET:
        cell.fov_edge = True
    elif (cell.center_x + cell.dimension_x) >= (fov_width - CELL_BOUNDARY_PIXEL_PRECISION_OFFSET):
        cell.fov_edge = True
    elif (cell.center_y + cell.dimension_y) >= (fov_height - CELL_BOUNDARY_PIXEL_PRECISION_OFFSET):
        cell.fov_edge = True
    else:
        cell.fov_edge = False

    return cell


def _make_cells(fov_id, region_props, pixel_unit_id, origin_id, fov, algorithm_id,
                run_id, segmentation_file_id, fov_width, fov_height):
    # Add the common information that we can derive from the FOV
    return [_make_cell(fov_id=fov_id,
                       region=region,
                       pixel_unit_id=pixel_unit_id,
                       origin_id=origin_id,
                       fov=fov,
                       algorithm_id=algorithm_id,
                       run_id=run_id,
                       segmentation_file_id=segmentation_file_id,
                       fov_width=fov_width,
                       fov_height=fov_height)
            for region in region_props]


def _get_or_create_cells(cells: CellList, lk_conn: LabKey = None) -> dict:
    # If cells is empty, that means the images we were given
    # had no cells in them. This can sometimes happen with
    # segmentation, so we'll warn and move on.
    if not len(cells):
        log.warning(f"No cells could be located in the images provided.")
        return {}

    # We'll put all cells in right now, and then remove
    # the cells that don't need to be added
    to_insert = cells.copy()

    cell_indices = [cell.cell_index for cell in cells]
    # These ids *should* all be the same
    fov_id = cells[0].fov_id
    run_id = cells[0].run_id
    algorithm_id = cells[0].content_generation_algorithm_id

    # Get all rows that match the unique keys provided
    # TODO: Use to_query_filters
    response = lk_conn.select_rows(
        schema_name=PROCESSING,
        query_name=CELL,
        filter_array=[
            QueryFilter("FOVId/FOVId", fov_id, "eq"),
            QueryFilter("CellIndex", ";".join(map(str, cell_indices)), "in"),
            QueryFilter("ContentGenerationAlgorithmId/ContentGenerationAlgorithmId", algorithm_id, "eq"),
            QueryFilter("RunId/RunId", run_id, "eq")
        ]
    )

    if response.get('rows') and len(response['rows']):
        # Given that FOV/Algorithm/Run *should* be the same, we can pop by cell index
        for row in response['rows']:
            this_index = row['CellIndex']
            log.info(f"Found cell record for cell index {this_index}")
            index = cell_indices.index(int(this_index))
            # Now we need to pop this from both the cell_indices and the to_insert
            cell_indices.pop(index)
            to_insert.pop(index)

    # Upload cells that actually need to be uploaded
    if len(to_insert):
        response = lk_conn.insert_rows(PROCESSING,
                                       CELL,
                                       rows=[cell.to_labkey() for cell in to_insert])
        if not response.get('rows') or not len(response['rows']):
            raise ValueError("Unable to enter Cell objects into the database: {}".format(response))
        for row in response['rows']:
            log.info(f"Inserted cell record for cell index {row['cellindex']}")

    # Lastly, we're just going to query Labkey for ALL the known cells
    # so the key access is consistent since inserts return all lowercase key :(
    response = lk_conn.select_rows(
        schema_name=PROCESSING,
        query_name=CELL,
        filter_array=[
            QueryFilter("FOVId/FOVId", fov_id, "eq"),
            QueryFilter("CellIndex", ";".join([str(cell.cell_index) for cell in cells]), "in"),
            QueryFilter("ContentGenerationAlgorithmId/ContentGenerationAlgorithmId", algorithm_id, "eq"),
            QueryFilter("RunId/RunId", run_id, "eq")
        ]
    )
    # If we don't get anything back from the query but we are expecting cells
    if not response.get('rows') or (not len(response['rows']) and len(cells)):
        raise ValueError("Unable to query for the expected cells: {}".format(response))
    # Create a CellId --> row map but remove all the detritus labkey puts in
    cell_id_map = {row['CellId']: {k: v for k, v in row.items() if not k.startswith('_labkey')}
                   for row in response['rows']}

    return cell_id_map


def _update_metadata(file_id: str, cell_ids: typing.List[int], fms: FileManagementSystem):
    delta = {'content_processing/cell_ids': cell_ids}
    log.debug("Updating file {} with cells {}".format(file_id, cell_ids))
    fms.uploader.update_metadata(file_id, delta)


def generate_cells(segmentation_file_path: str, segmentation_file_metadata: dict,
                   lk_conn: LabKey, update_metadata: bool = False) -> dict:
    """
    Generate Labkey Cell table entries from a file (+ its metadata) containing nucleus and membrane segmentations
    and contours

    Assumptions:
     - Input file contains 4 channels:
        [0] Nucleus segmentation
        [1] Membrane segmentation
        [2] Nucleus contour
        [3] Membrane contour
     - Metadata for input file contains all necessary information

    Steps:
     - Open input file
     - Parse info for each cell membrane in the input file, including bounding box dimensions and whether or not the
        cell makes contact with the edge of the FOV.
     - For each cell, generate a cell object in LabKey

    Returns:
        Mapping of cellid --> cell row information that is returned from Labkey.
        Note that all of the key names for the row information are lowercase because
        they are returned from postgres not Labkey.
    """
    log.info("Checking metadata for relevant information")
    _check_metadata_blocks(segmentation_file_metadata)

    # Get common information across all cells
    fov_id = segmentation_file_metadata['microscopy']['fov_id']
    log.debug(f"FOVId: {fov_id}")
    algorithm_id = segmentation_file_metadata['content_processing']['channels']['0']['algorithm_id']
    log.debug(f"ContentGenerationAlgorithmId: {algorithm_id}")

    # Run ID may or may not exist
    run_id = segmentation_file_metadata['content_processing']['channels']['0'].get('run_id')
    log.debug(f"RunId: {run_id}")

    segmentation_file_id = segmentation_file_metadata['file']['file_id']

    # Get entries from labkey
    fov = lk_conn.get_fov(fov_id)
    origin_id = _origin_id_from_name(ORIGIN, lk_conn)
    pixel_unit_id = _unit_id_from_name(PIXEL_UNIT, lk_conn)

    # Calculate the region properties and check the files
    log.info("Calculating region properties")
    memb_data = _get_image_data(segmentation_file_path)
    region_props = _get_region_properties(memb_data)
    fov_dimensions = memb_data.shape
    _, fov_height, fov_width = fov_dimensions

    # Make a cell object for each cell in the image
    log.info("Generating Cell entries for Labkey")
    cells = _make_cells(fov_id, region_props, pixel_unit_id, origin_id, fov, algorithm_id,
                        run_id, segmentation_file_id, fov_width, fov_height)

    # Records are made, now we need to insert them into the DB
    log.info("Inserting or retrieving Cell entries")
    cell_id_map = _get_or_create_cells(cells, lk_conn)
    log.debug(cell_id_map)

    # Add update the metadata if requested to and if cells were found
    if update_metadata and cell_id_map:
        fms = FileManagementSystem(lk_conn=lk_conn)
        cell_ids = list(cell_id_map.keys())
        log.info("Updating segmentation file metadata")
        _update_metadata(segmentation_file_id, cell_ids, fms)

    return cell_id_map


def generate_cells_from_fov_ids(fov_ids: DataFrame, lk: LabKey, init_time=datetime.utcnow().strftime('%Y%m%d_%H%M%S')):
    # Generate cells from a dataframe containing FOV info

    failed_fovs = DataFrame(columns=['FOV', 'Failure'])

    for index, row in fov_ids.iterrows():
        fovid = row['fovid']
        seg_readpath = row['latest_segmentation_readpath']
        seg_metadata = row['latest_segmentation_metadata']
        if seg_readpath is not None and seg_metadata is not None:
            try:
                log.info(f"Generating Cells for FOV {fovid}")
                log.debug(f"File path: {seg_readpath.strip()}")
                log.debug(f"Metadata:  {seg_metadata}")
                generate_cells(
                    segmentation_file_path=seg_readpath.strip(),
                    segmentation_file_metadata=json.loads(seg_metadata),
                    lk_conn=lk,
                    update_metadata=True
                )
            except Exception as e:
                # Continue through the loop if an exception occurs, but record info about the exception
                err_type = type(e).__name__  # Grab the exception type manually because str(e) doesn't include it
                err_msg = f"{err_type}: {e}"
                log.error(err_msg)
                failed_fovs = _update_failed_fovs(failed_fovs, fovid, err_msg, init_time)
        else:
            msg = 'File path or metadata block missing'
            log.info(f"Could not generate cells for FOV {fovid}; {msg}")
            failed_fovs = _update_failed_fovs(failed_fovs, fovid, msg, init_time)


def _update_failed_fovs(failed_fovs: DataFrame, fovid: str, failure_msg: str, init_time):
    failed_fovs = failed_fovs.append({'FOV': fovid, 'Failure': failure_msg},  ignore_index=True)
    failed_fovs.to_csv(f'cell_gen_cli_{init_time}_failed_fovs.csv')
    return failed_fovs
