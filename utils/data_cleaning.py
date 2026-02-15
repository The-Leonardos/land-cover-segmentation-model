# TODO: You may rename functions to your liking

class DataCleaning:
    def __init__(self):
        pass

    def clean(self, image, is_mask=False):
        # TODO: apply here the data cleaning pipeline for the mask or a satellite image
        #       NOTE: is_mask is used since there are some precesses that are only applicable for mask images and not for satellite images (e.g. irrelevant classes)
        #  - REMOVE PASS WHEN THIS FUNCTION IS DONE

        pass

    def remove_irrelevant_classes(self):
        # TODO: remove the class label
        #  - REMOVE PASS WHEN THIS FUNCTION IS DONE

        pass

    def remove_void_pixels(self):
        # TODO: not sure if this is needed?
        #  - REMOVE PASS WHEN THIS FUNCTION IS DONE

        pass

    def erode_boundaries(self):
        # TODO: apply boundary pixel erosion
        #  - REMOVE PASS WHEN THIS FUNCTION IS DONE

        pass

    def apply_mmp_filter(self):
        # TODO: apply minimum mapping unit filtering
        #  - REMOVE PASS WHEN THIS FUNCTION IS DONE

        pass