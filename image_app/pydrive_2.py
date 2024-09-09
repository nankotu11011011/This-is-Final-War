from pydrive2.drive import GoogleDrive
from pydrive2.auth import GoogleAuth
import os
import pandas as pd

class GoogleDriveAPI:
    def __init__(self):
        self.DIR_PATH = os.getcwd()
        self.gauth = GoogleAuth() # Google認証のインスタンス化
        self.gauth.LoadClientConfigFile(os.path.join(self.DIR_PATH + '\\image_app\\' + 'client_secrets.json'))
        self.drive = GoogleDrive(self.gauth) # GoogleDriveのインスタンス化
        self.gauth.LocalWebserverAuth()# ローカルWebサーバーで認証する


    def get_folder_id(self, folder_name):
        """フォルダ名からフォルダIDを取得"""
        query = "trashed = false and mimeType='application/vnd.google-apps.folder' and title='{}'".format(folder_name)
        folder_list = self.drive.ListFile({'q': query}).GetList()
        if len(folder_list) > 0:
            return folder_list[0]['id']
        else:
            return None
        
    
    def get_file_id(self, folder_name=None, file_name=None, folder_id=None):
        """フォルダ内の指定された名前の最新ファイルのファイルIDを取得する"""
        if not folder_id:
            folder_id = self.get_folder_id(folder_name)
        query = f"trashed = false and '{folder_id}' in parents and mimeType != 'application/vnd.google-apps.folder'"
        if file_name:
            query += f" and title = '{file_name}'"
            file_list = self.drive.ListFile({'q': query}).GetList()
        else:
            file_list = self.drive.ListFile({'q': query}).GetList()
            if len(file_list) > 0:
                latest_file = max(file_list, key=lambda f: f['createdDate'])
                return latest_file['id']
        if file_list:
            return file_list[0]['id']
        else:
            return None

    def get_all_file_id(self, folder_name):
        """フォルダー名からフォルダを特定し、そのフォルダ内のファイルID一覧を取得する"""
        folder_id = self.get_folder_id(folder_name)
        file_list = self.drive.ListFile({'q': f"\'{folder_id}\' in parents"}).GetList()
        file_id_list = []
        for file in file_list:
            file_id_list.append(file['id'])
        return file_id_list



        
    def get_file_df(self, folder_name):
        """フォルダ内のファイルIDと作成日時のデータフレームを取得する"""
        folder_id = self.get_folder_id(folder_name)
        query = f"trashed = false and '{folder_id}' in parents"
        file_list = self.drive.ListFile({'q': query}).GetList()
        file_id_list = []
        for file in file_list:
            file_id_list.append([file['title'] ,file['createdDate'] ,file['id']])
        file_df = pd.DataFrame(file_id_list, columns=['title', 'createdDate', 'id'])
        return file_df



    def get_created_at(self, folder_name):
        '''指定したフォルダ名のファイルIDと作成日時の辞書を取得する'''
        folder_id = self.get_folder_id(folder_name)
        query = f"trashed = false and '{folder_id}' in parents"
        file_list = self.drive.ListFile({'q': query}).GetList()
        file_dic = {}
        for i in range(len(file_list)):
            file_dic[file_list[i]['id']] = file_list[i]['createdDate']
        return file_dic

    def gdrive_upload(self, folder_id, content_file, file_name='no_name.csv'):
        '''Upload file to Google Drive'''
        metadata = {
            'parents':[
                {"id":folder_id}
            ],
            'title': file_name,
            'mimeType':'text/csv'
        }
        f = self.drive.CreateFile(metadata=metadata)
        f.SetContentFile(content_file)
        f.Upload()


    def gdrive_download(self, folder_name=None, output_name=None, file_id=None, target_dir=""):
        """ファイルを現在のディレクトリまたは指定されたディレクトリにダウンロードする"""
        if not file_id:
            folder_id = self.get_folder_id(folder_name)
            if not folder_id:
                print("Folder not found")
                return
            file_id = self.get_file_id(folder_id=folder_id)
            if not file_id:
                print("File not found")
                return
        f = self.drive.CreateFile({'id': file_id})
        f.FetchMetadata()
        if not f.metadata.get('trashed', False):
            f.GetContentFile(os.path.join(target_dir, output_name))

    def download_from_file_name(self, folder_name, file_name, target_dir=""):
        """指定したフォルダ名とファイル名から特定のファイルをダウンロードする"""
        folder_id = self.get_folder_id(folder_name)
        if not folder_id:
            print("Folder not found")
            return
        file_id = self.get_file_id(folder_name=folder_name, file_name=file_name)
        if not file_id:
            print("File not found")
            return
        f = self.drive.CreateFile({'id': file_id})
        f.FetchMetadata()
        if not f.metadata.get('trashed', False):
            f.GetContentFile(os.path.join(target_dir, file_name))