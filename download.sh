# proto='https://raw.githubusercontent.com/richzhang/colorization/master/models/colorization_deploy_v2.prototxt'
# model='https://people.eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2_norebal.caffemodel'

proto='https://raw.githubusercontent.com/richzhang/colorization/master/models/colorization_deploy_v2.prototxt'
model='http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel'

points='https://github.com/richzhang/colorization/raw/master/resources/pts_in_hull.npy'
pic='https://raw.githubusercontent.com/richzhang/colorization/master/demo/imgs/ansel_adams.jpg'
video='https://fpdl.vimeocdn.com/vimeo-prod-skyfire-std-us/01/2594/13/337972830/1342197457.mp4?token=1565544293-0x9c70c30b979b97ef3dc2fe9c6bae817eddf694d5'

wget -O 'colorization.prototxt' $proto
wget -O 'colorization.caffemodel' $model
wget -O 'points.npy' $points
wget -O 'test.jpg' $pic
wget -O 'test.mp4' $video
