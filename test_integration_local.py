import json
from app import app


def post_file(client, path):
    with open(path, 'rb') as f:
        data = {'file': (f, path)}
        resp = client.post('/predict', data=data, content_type='multipart/form-data')
        return resp.status_code, resp.get_data(as_text=True)


def main():
    client = app.test_client()
    # Representative MRI and non-MRI from the dataset
    mri = 'dataset_images/yes/y1.jpg'
    non_mri = 'dataset_images/no/no478.jpg'

    print('\nPosting MRI image:', mri)
    status, body = post_file(client, mri)
    print('Status:', status)
    print('Body:', body[:1000])

    print('\nPosting non-MRI image:', non_mri)
    status, body = post_file(client, non_mri)
    print('Status:', status)
    print('Body:', body[:1000])


if __name__ == '__main__':
    main()
