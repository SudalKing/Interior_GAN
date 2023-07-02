from torchvision import transforms
from options.test_options import TestOptions
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util

from PIL import Image
from flask import Flask, jsonify, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'datasets/interior_modern/testA/'


def model_test(model_name, model_dataroot, model_netG):
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.name = model_name
    opt.dataroot = model_dataroot
    opt.netG = model_netG
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    # create a webpage for viewing the results
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))


    for i, data in enumerate(dataset):
        if i == 0:
            model.data_dependent_initialize(data)
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            model.parallelize()
            if opt.eval:
                model.eval()
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        save_images(webpage, visuals, img_path, width=opt.display_winsize)
    webpage.save()  # save the HTML

def get_images_from_directory(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            images.append(filename)
    return images


def resize_and_save_image(image_path, output_path, target_size):
    image = Image.open(image_path)
    resized_image = image.resize(target_size)
    resized_image.save(output_path)


def get_images_and_save_to_static(source_dir, destination_dir, prefix=''):
    # source_dir에 있는 모든 파일을 가져옵니다.
    file_list = os.listdir(source_dir)

    # 각 파일을 순회하며 복사합니다.
    for file_name in file_list:
        source_path = os.path.join(source_dir, file_name)
        destination_path = os.path.join(destination_dir,prefix + file_name)
        # shutil.copy(source_path, destination_path)
        resize_and_save_image(source_path, destination_path, (400, 256))

    modified_file_list = [prefix + file_name for file_name in file_list]

    return modified_file_list


def get_images_and_save_to(source_dir, destination_dir):
    # source_dir에 있는 모든 파일을 가져옵니다.
    file_list = os.listdir(source_dir)

    # 각 파일을 순회하며 복사합니다.
    for file_name in file_list:
        source_path = os.path.join(source_dir, file_name)
        destination_path = os.path.join(destination_dir, file_name)
        image = Image.open(source_path)
        image.save(destination_path)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload_file():
    return render_template('1_upload.html')

@app.route('/uploader', methods=['GET', 'POST'])
def uploader_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        filename = secure_filename(f.filename)
        print("파일 이름: ", filename)

        # ============================ 딥러닝 처리 ================================
        modern_model_name = '2modern_CUT'
        modern_model_dataroot = './datasets/interior_modern'
        # model_test(modern_model_name, modern_model_dataroot, 'resnet_9blocks')

        natural_model_name = '2natural_CUT'
        natural_model_dataroot = './datasets/interior_natural'
        get_images_and_save_to(modern_model_dataroot + '/testA/', natural_model_dataroot + '/testA/')
        # model_test(natural_model_name, natural_model_dataroot, 'resnet_9blocks')

        white_model_name = '2white_CUT_2'
        # white_model_dataroot = './datasets/interior_white'
        # get_images_and_save_to(modern_model_dataroot + '/testA/', white_model_dataroot + '/testA/')
        # model_test(white_model_name, white_model_dataroot, 'stylegan2')
        # ===================================================================================
        # ===========================이미지 경로 처리=====================================
        base_source_dir = 'datasets/interior_modern/testA/'
        destination_dir = 'static/images/'
        get_images_and_save_to_static(base_source_dir, destination_dir, 'base_')
        
        modern_dir = 'results/' + modern_model_name + '/test_latest/images/fake_B/'
        get_images_and_save_to_static(modern_dir, destination_dir, 'modern_')

        natural_dir = 'results/' + natural_model_name + '/test_latest/images/fake_B/'
        get_images_and_save_to_static(natural_dir, destination_dir, 'natural_')

        white_dir = 'results/' + white_model_name + '/test_latest/images/fake_B/'
        get_images_and_save_to_static(white_dir, destination_dir, 'white_')
        # ===============================================================================

        image_list = []
        image_list = get_images_from_directory(destination_dir)
        print(image_list)

        return render_template('2_tmp.html', image_list=image_list)
    
    return 'file upload failed'

if __name__ == '__main__':
    app.run(debug=True)