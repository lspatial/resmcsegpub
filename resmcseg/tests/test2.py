

from google_drive_downloader import GoogleDriveDownloader as gdd

gdd.download_file_from_google_drive(file_id='1ptvGriYgU49AJMFiu8CPcS51edsFRcbG',
                                    dest_path='/tmp/pretrained.h5',
                                    unzip=True)