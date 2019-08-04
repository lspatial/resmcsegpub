from google_drive_downloader import GoogleDriveDownloader as gdd

def downloadPretrainedModel(dataset,type=None,destination=None):
   """
    Function to download the pretrained models from Google cloud platform
      :param dataset: only trained models can be used by two dataset, i.e., DSTL and Zurich;
      :param type: model for the type of segmentation, useful for binary semantic segmentation of DSTL,
              Buildings, Trees,Vehicles, Crops and Roads;
              Models trained for the following data preprocessing:
                  DSTL: bands order: RGB, M, A, P (all bands used), A normalized by dividing by 16384,
                         the others normalized by dividing by 2048;
                  ZURICH: band orders: NIR-R-G-B, normalzied by dividing by 2026
      :param destinationFile: the destination file to be saved, default will be in the tmp path!
      :return: None
    """
   urlMapping={'DSTL':{'Trees':'1nWcHxlrSbpt6NhtOyRuoDxuKYL6Wi26d','Buildings':'1SjjlYPqGc1dICWb2o2wk40N5fgumdbns',
                       'Crops':'1ULlFc5mD8G9b6o1tdy8OjlNZiU0FZG5m','Roads':'1s1O8N5rnPrm7aupTEX2oZkFrTCm3dptg',
                       'Vehicles':'1ptvGriYgU49AJMFiu8CPcS51edsFRcbG'}, 'ZURICH':'1JXfnkfNNxHWsvYbGYQg_NmhP4hJggAW7'}
   if dataset not in urlMapping.keys():
       print("Only pretrained models for DSTL and ZURICH available!!!")
       return
   file_id=None
   if dataset=='ZURICH':
       if destination is None:
           destination='/tmp/ZURICH_model.h5'
       file_id=urlMapping[dataset]
   elif dataset=='DSTL':
       if type is None or type not in urlMapping[dataset].keys():
           print("Only accepting the types of Buildings, Trees, Vehicles, Crops and Roads for DSTL e!!!")
           return
       else:
           if destination is None:
               destination='/tmp/DSLT_'+type+'_model.h5'
           file_id=urlMapping[dataset][type]
   else:
       return
   if file_id is not None:
       gdd.download_file_from_google_drive(file_id=file_id,dest_path=destination,unzip=False)
