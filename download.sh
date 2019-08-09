proto='https://raw.githubusercontent.com/richzhang/colorization/master/models/colorization_deploy_v2.prototxt'
model='http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel'
points='https://github.com/richzhang/colorization/raw/master/resources/pts_in_hull.npy'
pic='https://raw.githubusercontent.com/richzhang/colorization/master/demo/imgs/ansel_adams.jpg'

wget -O 'colorization.prototxt' $proto
wget -O 'colorization.caffemodel' $model
wget -O 'points.npy' $points
wget -O 'test.jpg' $pic
