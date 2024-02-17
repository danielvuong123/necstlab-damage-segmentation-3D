import os
import shutil
import random
from skimage import io
import numpy as np
from tensorflow import random as tf_random
import yaml
from datetime import datetime
import pytz
from PIL import Image, ImageOps
from pathlib import Path
import git
from models import generate_compiled_segmentation_model
from image_utils import str2bool
from metrics_utils import global_threshold
from local_utils import local_folder_has_files, getSystemInfo, getLibVersions
from patchify import patchify, unpatchify
import tifftools
import tifffile as tif
from tqdm import tqdm



# infer can be run multiple times (labels, overlay), create new metadata each time
infer_datetime = datetime.now(pytz.UTC).strftime('%Y%m%dT%H%M%SZ')
metadata_file_name = 'metadata_' + infer_datetime + '.yaml'

tmp_directory = Path('./tmp')

# rgb
class_colors = [
    [0, 0, 255],    # blue
    [255, 255, 0],  # yellow
    [255, 0, 0],    # red
    [0, 255, 0],    # green
    [255, 0, 255]   # magenta
]


def stitch_preds_together(tiles, target_size_1d, labels_output, pad_output, image):

    #Once again, deal with the 3D case on it's own:
    if target_size_1d <= 64:
        #Again, the output MUST be padded in the 3D case, so we remove the padding.
        padded_image_size = target_size_1d * np.ceil(np.asarray(image.shape) / target_size_1d).astype(int)
        volume = unpatchify(tiles,padded_image_size) #Should be of shape of
        delta_w = padded_image_size[0] - image.shape[0]
        delta_l = padded_image_size[1] - image.shape[1]
        delta_h = padded_image_size[2] - image.shape[2]

        #Slice away the padding, keeping the orignal image.
        predictions_without_padding = volume[(delta_w // 2)+(1*delta_w):(delta_w // 2)+1+image.shape[0], (delta_l // 2)+1:(delta_l // 2)+1+image.shape[1], (delta_h // 2)+1:(delta_h // 2)+1+image.shape[2]]

        #Ensure that we have the correct dimensions for the predictions
        assert predictions_without_padding.shape == image.shape

        return (predictions_without_padding*255).astype('uint8')

    n_tile_rows = len(tiles)
    n_tile_cols = len(tiles[0])
    if not pad_output:
        stitched_array = np.zeros((image.size[1], image.size[0], 3))
    else:
        stitched_array = np.zeros((target_size_1d * n_tile_rows, target_size_1d * n_tile_cols, 3))
    for i in range(n_tile_rows):
        for j in range(n_tile_cols):
            if not pad_output and i == n_tile_rows - 1 and j == n_tile_cols - 1:
                stitched_array[image.size[1] - target_size_1d:image.size[1], image.size[0] - target_size_1d:image.size[0], :] = tiles[i][j]
            elif not pad_output and i == n_tile_rows - 1:
                stitched_array[image.size[1] - target_size_1d:image.size[1], j * target_size_1d:(j + 1) * target_size_1d, :] = tiles[i][j]
            elif not pad_output and j == n_tile_cols - 1:
                stitched_array[i * target_size_1d:(i + 1) * target_size_1d, image.size[0] - target_size_1d:image.size[0], :] = tiles[i][j]
            else:
                stitched_array[i * target_size_1d:(i + 1) * target_size_1d, j * target_size_1d:(j + 1) * target_size_1d, :] = tiles[i][j]

    if labels_output:
        stitched_image = Image.fromarray(np.mean(stitched_array, -1).astype('uint8'))
    else:
        stitched_image = Image.fromarray(stitched_array.astype('uint8'))
    return stitched_image

def prepare_image_3D(image, target_size_1d, pad_output):
    '''
    image: array that is WxHxD representing our image.
    target_size_1d: The new desired size of the image
    pad_output: Placeholder variable

    Returns: In this case, a (x,y,z,W,H,D,3) shape output.

    Our model only accepts inputs of multiples of 64, however our input images
    do not come in this size by default. We pad the image in order to obtain
    the desired size.


    '''
    desired_size = target_size_1d * np.ceil(np.asarray(image.shape) / target_size_1d).astype(int)
    delta_w = desired_size[0] - image.shape[0]
    delta_l = desired_size[1] - image.shape[1]
    delta_h = desired_size[2] - image.shape[2]

    #Images MUST be padded for 3D
    paddingw = (delta_w // 2,delta_w - (delta_w // 2))
    paddingl = (delta_l // 2,delta_l - (delta_l // 2))
    paddingh = (delta_h // 2,delta_h - (delta_h // 2))
    padding = (paddingw,paddingl,paddingh)

    #Pad the image on all sides with the average of the pixels
    padded_image = np.pad(image,padding,'mean')

    tiles = patchify(padded_image,(target_size_1d,)*3,step = target_size_1d)
    shape_3D = tiles.shape
    tiles = tiles * 1./255

    #Assert that the minimum at this point is zero
    #and the that maximum is 1
    assert np.min(tiles) >= 0
    assert np.max(tiles) <= 1

    return tiles

def prepare_image(image, target_size_1d, pad_output):
    #If the targetsize is larger than 2D return the 3D version instead.
    if target_size_1d <= 64:
        return prepare_image_3D(image, target_size_1d, pad_output)

    # make the image an event multiple of 512x512
    desired_size = target_size_1d * np.ceil(np.asarray(image.size) / target_size_1d).astype(int)
    delta_w = desired_size[0] - image.size[0]
    delta_h = desired_size[1] - image.size[1]
    if pad_output:
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    else:
        padding = (0, 0, 0, 0)
    padded_image = ImageOps.expand(image, padding, fill=int(np.asarray(image).mean()))

    # break into 512x512 tiles
    padded_image = np.asarray(padded_image)
    tiles = []
    for i in range(np.ceil(padded_image.shape[0] / target_size_1d).astype(int)):
        tiles.append([])
        for j in range(np.ceil(padded_image.shape[1] / target_size_1d).astype(int)):
            if (not pad_output and i == np.ceil(padded_image.shape[0] / target_size_1d).astype(int) - 1
                    and j == np.ceil(padded_image.shape[1] / target_size_1d).astype(int) - 1):
                tiles[i].append(padded_image[image.size[1] - target_size_1d:image.size[1],
                                             image.size[0] - target_size_1d:image.size[0]].copy())
            elif not pad_output and i == np.ceil(padded_image.shape[0] / target_size_1d).astype(int) - 1:
                tiles[i].append(padded_image[image.size[1] - target_size_1d:image.size[1],
                                             j * target_size_1d:(j + 1) * target_size_1d].copy())
            elif not pad_output and j == np.ceil(padded_image.shape[1] / target_size_1d).astype(int) - 1:
                tiles[i].append(padded_image[i * target_size_1d:(i + 1) * target_size_1d,
                                             image.size[0] - target_size_1d:image.size[0]].copy())
            else:
                tiles[i].append(padded_image[i * target_size_1d:(i + 1) * target_size_1d,
                                             j * target_size_1d:(j + 1) * target_size_1d].copy())

    # scale the images to be between 0 and 1 if GV
    for i in range(len(tiles)):
        for j in range(len(tiles[i])):
            tiles[i][j] = tiles[i][j] * 1. / 255
    return tiles


def overlay_predictions(prepared_tiles, preds, prediction_threshold, background_class_index, labels_output):

    #Again, deal with the 3D case on it's own:
    if not isinstance(preds,list):
        assert prepared_tiles.shape == preds.shape

        if labels_output:
            prediction_tiles = np.where(preds > 0,preds,0)
        else:
            prediction_tiles = np.where(preds > 0,preds,prediction_tiles)
            final_shape = prepared_tiles.shape
            np.zeros(prediction_tiles.shape)
            for i in range(len(prepared_tiles)):
                for j in range(len(prepared_tiles[i])):
                    for k in range(len(prepared_tiles[j])):
                        prediction_tiles[i,j,k] = np.dstack((prepared_tiles,)*3)
                        prediction_tiles[i,j,k] = (prediction_tiles[i,j,k] * 255).astype(int)

                        ####CURRENTLY ONLY WORKS WITH SINGLE CLASS, WORK ON MULTICLASS####

        return prediction_tiles


    prediction_tiles = []
    for i in range(len(prepared_tiles)):
        prediction_tiles.append([])
        for j in range(len(prepared_tiles[i])):
            #Make a 3 channel prediction value
            prediction_tiles[i].append(np.dstack((prepared_tiles[i][j], prepared_tiles[i][j], prepared_tiles[i][j])))
            prediction_tiles[i][j] = (prediction_tiles[i][j] * 255).astype(int)

            relative_above_threshold_mask = np.divide(preds[i][j], np.multiply(np.ones_like(preds[i][j]),
                                                                               prediction_threshold)).max(axis=-1) > 1
            best_class_by_pixel = np.divide(preds[i][j], np.multiply(np.ones_like(preds[i][j]),
                                                                     prediction_threshold)).argmax(axis=-1)
            color_counter = 0
            for class_i in range(preds[i][j].shape[-1]):
                rel_above_threshold_and_best_class = relative_above_threshold_mask & (best_class_by_pixel == class_i)
                if (background_class_index is not None) and (class_i == background_class_index):
                    continue
                if labels_output:
                    prediction_tiles[i][j][rel_above_threshold_and_best_class] = int((color_counter + 1) *
                                                                                     np.floor(255 / preds[i][j].shape[-1]))
                else:
                    prediction_tiles[i][j][rel_above_threshold_and_best_class] = class_colors[color_counter]
                color_counter = (color_counter + 1) % len(class_colors)
    return prediction_tiles


def segment_image(model, image, prediction_threshold, target_size_1d, background_class_index,
                  labels_output, pad_output):

    #In the 3D case this returns a (x,y,z,W,H,D) output
    prepared_tiles = prepare_image(image, target_size_1d, pad_output)

    #Deal with 3D case on it's own
    if target_size_1d <= 64:
        #Start with a empty array of the desired shape.
        preds = np.zeros(prepared_tiles.shape)
        print(prepared_tiles.shape)
        predicted_3D_patches = []
        for i in range(prepared_tiles.shape[0]):
            for j in range(prepared_tiles.shape[1]):
                for k in range(prepared_tiles.shape[2]):
                    single_patch = prepared_tiles[i,j,k,:,:,:]
                    prediction = model.predict(np.expand_dims(single_patch,axis=0))
                    prediction_rounded = np.round(prediction)[0,:,:,:,0] #Ignoreing thresholding
                    predicted_3D_patches.append(prediction_rounded) ####Temporary Analysis####
                    #preds[i,j,k,:,:,:] = prediction_rounded #prediction is now 64x64x64

        #TEMPORARY FIX: Just return the labels, ignore everything else.
        predicted_patches_3D = np.array(predicted_3D_patches)
        predicted_patches_3D_reshaped = np.reshape(predicted_patches_3D,prepared_tiles.shape)
        desired_size = target_size_1d * np.ceil(np.asarray(image.shape) / target_size_1d).astype(int)
        reconstructed_image_3D = unpatchify(predicted_patches_3D_reshaped,desired_size)
        delta_w = desired_size[0] - image.shape[0]
        delta_l = desired_size[1] - image.shape[1]
        delta_h = desired_size[2] - image.shape[2]

        if delta_w == 0:
            predictions_without_padding = reconstructed_image_3D[(delta_w // 2):(delta_w // 2)+image.shape[0], (delta_l // 2):(delta_l // 2)+image.shape[1], (delta_h // 2):(delta_h // 2)+image.shape[2]]
        else:
            predictions_without_padding = reconstructed_image_3D[(delta_w // 2):(delta_w // 2)+image.shape[0], (delta_l // 2):(delta_l // 2)+image.shape[1], (delta_h // 2):(delta_h // 2)+image.shape[2]]

        assert predictions_without_padding.shape == image.shape

        return (predictions_without_padding*255).astype('uint8')
        #End TEMPORARY code.

        #Prepared_tiles are the raw images, we blank them out
        #if we are only looking at the labels output
        if labels_output:
            prepared_tiles = np.zeros(prepared_tiles.shape)
        pred_tiles = overlay_predictions(prepared_tiles, preds, prediction_threshold, background_class_index, labels_output)
        stitched_pred = stitch_preds_together(pred_tiles, target_size_1d, labels_output, pad_output, image)
        return stitched_pred

    preds = []
    for i in range(len(prepared_tiles)):
        preds.append([])
        for j in range(len(prepared_tiles[i])):
            #prepared_tiles = 512x512, reshape makes it 1x512x512x1,
            #Fetching the first index of the prediction then gives us a 512,512,1 3d output.
            preds[i].append(model.predict(prepared_tiles[i][j].reshape(1, target_size_1d,
                                                                       target_size_1d, 1))[0, :, :, :])

    # make background black if labels only
    if labels_output:
        for i in range(len(prepared_tiles)):
            for j in range(len(prepared_tiles[i])):
                prepared_tiles[i][j] = prepared_tiles[i][j] * 0

    pred_tiles = overlay_predictions(prepared_tiles, preds, prediction_threshold, background_class_index, labels_output)
    stitched_pred = stitch_preds_together(pred_tiles, target_size_1d, labels_output, pad_output, image)
    return stitched_pred


def merge_tifs(directory,input_tifs,name="tmp"):
    """
    directory:
        directory where the images are stored.
    input_tifs:
        a list of tif images to be merged
    name:
        name of the output file
    """
    image = tifftools.read_tiff(input_tifs[0])
    for input in input_tifs[1:]:
        next_image = tifftools.read_tiff(input)
        image['ifds'].extend(next_image['ifds'])

    tifftools.write_tiff(image,directory.as_posix() + "/" + name + ".tif")
    return directory.as_posix() + "/" + name + ".tif"

def merge_directory_tif(directory,name="tmp"):
    """
    directory (str):
        name of directory with .tif images to be merged into a singlular tif file
    name (str):
        name of the compiled tif image, defaulted to "temp"

    returns (str):
        Name of the file created
    """
    image = None
    count =0
    for filename in sorted(os.listdir(directory)):
        count+=1
        # print(count)
        if image is None:
            image = tifftools.read_tiff(os.path.join(directory, filename))
        else:
            next_image = tifftools.read_tiff(os.path.join(directory, filename))
            image['ifds'].extend(next_image['ifds'])

    tifftools.write_tiff(image,directory.as_posix() + "/" + name + ".tif")
    return directory.as_posix() + "/" + name + ".tif"

def main(gcp_bucket, model_id, background_class_index, stack_id, image_ids, user_specified_prediction_thresholds,
         labels_output, pad_output, trained_thresholds_id, random_module_global_seed, numpy_random_global_seed,
         tf_random_global_seed, message):

    # seed global random generators if specified; global random seeds here must be int or default None (no seed given)
    if random_module_global_seed is not None:
        random.seed(random_module_global_seed)
    if numpy_random_global_seed is not None:
        np.random.seed(numpy_random_global_seed)
    if tf_random_global_seed is not None:
        tf_random.set_seed(tf_random_global_seed)

    start_dt = datetime.now()

    assert "gs://" in gcp_bucket

    if background_class_index is not None:
        assert background_class_index >= 0

    # clean up the tmp directory
    try:
        shutil.rmtree(tmp_directory.as_posix())
    except FileNotFoundError:
        pass
    tmp_directory.mkdir()

    run_name = '{}_{}'.format(stack_id, model_id)

    local_model_dir = Path(tmp_directory, 'models', model_id)
    local_model_dir.mkdir(parents=True)
    local_processed_data_dir = Path(tmp_directory, 'processed-data', stack_id)
    local_processed_data_dir.mkdir(parents=True)
    local_inferences_dir = Path(tmp_directory, 'inferences', run_name)
    local_inferences_dir.mkdir(parents=True)
    output_dir = Path(local_inferences_dir, str('output_' + infer_datetime))
    output_dir.mkdir(parents=True)

    os.system("gsutil -m cp -r '{}' '{}'".format(os.path.join(gcp_bucket, 'models', model_id),
                                                 Path(tmp_directory, 'models').as_posix()))

    local_folder_has_files(local_model_dir, model_id)

    os.system("gsutil -m cp -r '{}' '{}'".format(os.path.join(gcp_bucket, 'processed-data', stack_id),
                                                 Path(tmp_directory, 'processed-data').as_posix()))

    local_folder_has_files(local_processed_data_dir, stack_id)

    with Path(local_model_dir, 'config.yaml').open('r') as f:
        train_config = yaml.safe_load(f)['train_config']

    with Path(local_model_dir, 'metadata.yaml').open('r') as f:
        model_metadata = yaml.safe_load(f)

    if trained_thresholds_id is not None:
        with Path(local_model_dir, trained_thresholds_id).open('r') as f:
            threshold_output_data = yaml.safe_load(f)

    image_folder = Path(local_processed_data_dir, 'images')
    assert model_metadata['target_size'][0] == model_metadata['target_size'][1]
    target_size_1d = model_metadata['target_size'][0]
    num_classes = model_metadata['num_classes']

    optimized_class_thresholds = {}
    if trained_thresholds_id is not None and 'thresholds_training_output' in threshold_output_data['metadata']:
        for i in range(num_classes):
            if ('x' in threshold_output_data['metadata']['thresholds_training_output'][str('class' + str(i))] and
                    threshold_output_data['metadata']['thresholds_training_output'][str('class' + str(i))]['success']):
                optimized_class_thresholds.update(
                    {str('class' + str(i)): threshold_output_data['metadata']['thresholds_training_output'][str('class' + str(i))]['x']}
                )
            else:
                AssertionError('Unsuccessfully trained threshold attempted to be loaded.')
    else:
        optimized_class_thresholds = None

    # set threshold(s) used for inference
    if user_specified_prediction_thresholds:
        if len(user_specified_prediction_thresholds) == 1:
            prediction_threshold = np.ones(num_classes) * user_specified_prediction_thresholds
        else:
            assert len(user_specified_prediction_thresholds) == num_classes
            prediction_threshold = np.asarray(user_specified_prediction_thresholds)
    elif trained_thresholds_id is not None and 'thresholds_training_output' in threshold_output_data['metadata']:
        prediction_threshold = np.empty(num_classes)
        for i in range(num_classes):
            if ('x' in threshold_output_data['metadata']['thresholds_training_output'][str('class' + str(i))] and
                    threshold_output_data['metadata']['thresholds_training_output'][str('class' + str(i))]['success']):
                prediction_threshold[i] = threshold_output_data['metadata']['thresholds_training_output'][str('class' + str(i))]['x']
            else:
                AssertionError('Unsuccessfully trained threshold attempted to be loaded.')
    else:
        prediction_threshold = np.ones(num_classes) * global_threshold

    compiled_model = generate_compiled_segmentation_model(
        train_config['segmentation_model']['model_name'],
        train_config['segmentation_model']['model_parameters'],
        num_classes,
        train_config['loss'],
        train_config['optimizer'],
        Path(local_model_dir, "model.hdf5").as_posix(),
        optimized_class_thresholds=optimized_class_thresholds)

    if image_ids is None:
        images_list = []
        for i in Path(image_folder).iterdir():
            images_list.append(i.parts[-1])
    else:
        images_list = image_ids.split(',')

    labels_output = str2bool(labels_output)

    pad_output = str2bool(pad_output)

    n_images = len(list(Path(image_folder).iterdir()))

    if len(model_metadata['target_size']) > 2:
        # full_stack_image = merge_directory_tif(Path(image_folder),'full_stack_image')
        # full_stack_image = io.imread(full_stack_image)
        # segmented_image = segment_image(compiled_model,full_stack_image,prediction_threshold,
        #                     target_size_1d, background_class_index,labels_output,pad_output)
        # for i in range(n_images):
        #     tif.imsave(output_dir + "inference" + str(i) + '.tif', segmented_image[i,:,:])
        stacked_segmented_images = []
        index = 0
        for i in range(len(sorted(Path(image_folder).iterdir()))//64):
            current_stack = sorted(Path(image_folder).iterdir())[64*i:i*64+64]
            stacked_image = merge_tifs(Path(image_folder),current_stack,f"stack_{i}")
            stacked_image = io.imread(stacked_image)
            segmented_image = segment_image(compiled_model,stacked_image,prediction_threshold,
                                            target_size_1d, background_class_index,labels_output,pad_output)
            for j in range(segmented_image.shape[0]):
                tif.imsave(output_dir.as_posix() + f"inference_{index:04}" + ".tif", segmented_image[j,:,:])
                index += 1

        # for segmented_stack in stacked_segmented_images:
        #     for i in range(segmented_stack.shape[0]):
        #         tif.imsave(output_dir.as_posix() + f"inference_{index}" + ".tif",segmented_stack[i,:,:])
        #         index +=1


    for i, image_file in enumerate(sorted(Path(image_folder).iterdir())):
        #Don't go into this loop if we have a 3D input
        if len(model_metadata['target_size']) > 2:
            break
        if image_file.parts[-1] in images_list:
            print('Segmenting image {} --- stack has {} images...'.format(image_file.parts[-1], n_images))

            image = Image.open(image_file)

            segmented_image = segment_image(compiled_model, image, prediction_threshold,
                                            target_size_1d, background_class_index, labels_output, pad_output)

            # enable saving of various versions of same inference
            image_file_ext = image_file.parts[-1].split('.')[-1]
            if labels_output and pad_output:
                segmented_image.save(Path(output_dir, str(
                    image_file.parts[-1].split('.')[0] + '_pad_labels' + '.'
                    + image_file_ext)).as_posix())
            elif labels_output:
                segmented_image.save(Path(output_dir, str(
                    image_file.parts[-1].split('.')[0] + '_labels' + '.'
                    + image_file_ext)).as_posix())
            elif pad_output:
                segmented_image.save(Path(output_dir, str(
                    image_file.parts[-1].split('.')[0] + '_pad' + '.'
                    + image_file_ext)).as_posix())
            else:
                segmented_image.save(Path(output_dir, str(
                    image_file.parts[-1].split('.')[0] + '.'
                    + image_file_ext)).as_posix())

    metadata_sys = {
        'System_info': getSystemInfo(),
        'Lib_versions_info': getLibVersions()
    }

    metadata = {
        'message': message,
        'gcp_bucket': gcp_bucket,
        'model_id': model_id,
        'user_specified_prediction_thresholds': user_specified_prediction_thresholds,
        'trained_thresholds_id': trained_thresholds_id,
        'trained_class_thresholds_loaded': optimized_class_thresholds,
        'default_global_threshold_for_reference': global_threshold,
        'prediction_thresholds_used': prediction_threshold.tolist(),
        'background_class_index': background_class_index,
        'stack_id': stack_id,
        'image_ids': image_ids,
        'labels_output': labels_output,
        'pad_output': pad_output,
        'created_datetime': datetime.now(pytz.UTC).strftime('%Y%m%dT%H%M%SZ'),
        'git_hash': git.Repo(search_parent_directories=True).head.object.hexsha,
        'elapsed_minutes': round((datetime.now() - start_dt).total_seconds() / 60, 1),
        'random-module-global-seed': random_module_global_seed,
        'numpy_random_global_seed': numpy_random_global_seed,
        'tf_random_global_seed': tf_random_global_seed,
        'metadata_system': metadata_sys
    }

    with Path(local_inferences_dir, metadata_file_name).open('w') as f:
        yaml.safe_dump(metadata, f)

    os.system("gsutil -m cp -n -r '{}' '{}'".format(Path(tmp_directory, 'inferences').as_posix(), gcp_bucket))

    print('\n Infer Metadata:')
    print(metadata)
    print('\n')

    shutil.rmtree(tmp_directory.as_posix())


if __name__ == "__main__":
    import argparse
    import sys

    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument(
        '--gcp-bucket',
        type=str,
        help='The GCP bucket where the raw data is located and to use to store the processed stacks.')
    argparser.add_argument(
        '--model-id',
        type=str,
        help='The model ID.')
    argparser.add_argument(
        '--background-class-index',
        type=int,
        default=None,
        help='For this model, indicate background class index if used during model training, to exclude background overlay.')
    argparser.add_argument(
        '--stack-id',
        type=str,
        help='The stack ID (must already be processed).')
    argparser.add_argument(
        '--image-ids',
        type=str,
        default=None,
        help='For these images, the corresponding stack ID (must already be processed).')
    argparser.add_argument(
        '--user-specified-prediction-thresholds',
        type=float,
        nargs='+',
        default=None,
        help='Threshold(s) to apply to the prediction to classify a pixel as part of a class. E.g., 0.5 or 0.5 0.3 0.6')
    argparser.add_argument(
        '--labels-output',
        type=str,
        default='False',
        help='If false, will output overlaid image (RGB); if true, will output labels only image (GV).')
    argparser.add_argument(
        '--pad-output',
        type=str,
        default='False',
        help='If false, will output inference identical to input image size.')
    argparser.add_argument(
        '--trained-thresholds-id',
        type=str,
        default=None,
        help='The specified trained thresholds file id.')
    argparser.add_argument(
        '--random-module-global-seed',
        type=int,
        default=None,
        help='The setting of random.seed(global seed), where global seed is int or default None (no seed given).')
    argparser.add_argument(
        '--numpy-random-global-seed',
        type=int,
        default=None,
        help='The setting of np.random.seed(global seed), where global seed is int or default None (no seed given).')
    argparser.add_argument(
        '--tf-random-global-seed',
        type=int,
        default=None,
        help='The setting of tf.random.set_seed(global seed), where global seed is int or default None (no seed given).')
    argparser.add_argument(
        '--message',
        type=str,
        default=None,
        help='A str message the used wants to leave, the default is None.')
    main(**argparser.parse_args().__dict__)
