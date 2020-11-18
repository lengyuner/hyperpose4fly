
import os
import tensorlayer as tl
from tensorlayer import logging
from tensorlayer.files.utils import (del_file, folder_exists, maybe_download_and_extract)

from ..common import unzip

def prepare_dataset(data_path="./data",version="2020",task="person"):
        """Download MSCOCO Dataset.
        Both 2014 and 2017 dataset have train, validate and test sets, but 2017 version put less data into the validation set (115k train, 5k validate) i.e. has more training data.

        Parameters
        -----------
        path : str
            The path that the data is downloaded to, defaults is ``data/mscoco...``.
        dataset : str
            The MSCOCO dataset version, `2014` or `2017`.
        task : str
            person for pose estimation, caption for image captioning, instance for segmentation.

        Returns
        ---------
        train_im_path : str
            Folder path of all training images.
        train_ann_path : str
            File path of training annotations.
        val_im_path : str
            Folder path of all validating images.
        val_ann_path : str
            File path of validating annotations.
        test_im_path : str
            Folder path of all testing images.
        test_ann_path : None
            File path of testing annotations, but as the test sets of MSCOCO 2014 and 2017 do not have annotation, returns None.

        Examples
        ----------
        >>> train_im_path, train_ann_path, val_im_path, val_ann_path, _, _ = \
        ...    tl.files.load_mscoco_dataset('data', '2017')

        References
        -------------
        - `MSCOCO <http://mscoco.org>`__.

        """

        if version == "2020":
            # 11.5w train, 0.5w valid, test (no annotation)
            path = os.path.join(data_path, 'fly')

            if folder_exists(os.path.join(path, "annotations")) is False:
                logging.info("    annotations not exists")
                # logging.info("    downloading annotations")
                # os.system("wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P {}".format(path))
                # unzip(os.path.join(path, "annotations_trainval2017.zip"), path)
                # del_file(os.path.join(path, "annotations_trainval2017.zip"))
            else:
                logging.info("    annotations exists")

            if folder_exists(os.path.join(path, "val2017")) is False:
                # logging.info("    downloading validating images")
                # os.system("wget http://images.cocodataset.org/zips/val2017.zip -P {}".format(path))
                # unzip(os.path.join(path, "val2017.zip"), path)
                # del_file(os.path.join(path, "val2017.zip"))
                logging.info("    validating images not exists")
            else:
                logging.info("    validating images exists")
            
            if folder_exists(os.path.join(path, "train2017")) is False:
                # logging.info("    downloading training images")
                # os.system("wget http://images.cocodataset.org/zips/train2017.zip -P {}".format(path))
                # unzip(os.path.join(path, "train2017.zip"), path)
                # del_file(os.path.join(path, "train2017.zip"))
                logging.info("    training images not exists")
            else:
                logging.info("    training images exists")
            
            '''
            if folder_exists(os.path.join(path, "test2017")) is False:
                logging.info("    downloading testing images")
                os.system("wget http://images.cocodataset.org/zips/test2017.zip -P {}".format(path))
                unzip(os.path.join(path, "test2017.zip"), path)
                del_file(os.path.join(path, "test2017.zip"))
            
            else:
                logging.info("    testing images exists")
            '''
            print("temply ignore test dataset")
        else:
            raise Exception("dataset can only be 2020, see our website for more details. d@gmail.com")

        if version == "2020":
            train_images_path = os.path.join(path, "train2020")
            if task == "person":
                train_annotations_file_path = os.path.join(path, "annotations", "fly_keypoints_train2020.json")
            elif task == "caption":
                train_annotations_file_path = os.path.join(path, "annotations", "captions_train2017.json")
            elif task == "instance":
                train_annotations_file_path = os.path.join(path, "annotations", "instances_train2017.json")
            else:
                raise Exception("unknown task")
            val_images_path = os.path.join(path, "val2020")
            if task == "person":
                val_annotations_file_path = os.path.join(path, "annotations", "fly_keypoints_val2020.json")
            elif task == "caption":
                val_annotations_file_path = os.path.join(path, "annotations", "captions_val2017.json")
            elif task == "instance":
                val_annotations_file_path = os.path.join(path, "annotations", "instances_val2017.json")
            test_images_path = os.path.join(path, "test2020")
            test_annotations_file_path =None # os.path.join(path, "annotations", "person_keypoints_test2017.json")

        return train_images_path,train_annotations_file_path,\
                     val_images_path,val_annotations_file_path,\
                        test_images_path,test_annotations_file_path
        