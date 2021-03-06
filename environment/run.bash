XAUTH=/tmp/.docker.xauth
if [ ! -f $XAUTH ]
then
    xauth_list=$(xauth nlist :0 | sed -e 's/^..../ffff/')
    if [ ! -z "$xauth_list" ]
    then
        echo $xauth_list | xauth -f $XAUTH nmerge -
    else
        touch $XAUTH
    fi
    chmod a+r $XAUTH
fi

PARENTDIR="$(dirname $PWD)"

xhost +

docker run -it --rm \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --env="XAUTHORITY=$XAUTH" \
    --volume="$XAUTH:$XAUTH" \
    --volume="$PARENTDIR/src/catkin_ws:/root/catkin_ws/" \
    --volume="$PARENTDIR/data:/root/data/" \
    --volume="$PARENTDIR/.gazebo_models/:/root/.gazebo/models" \
    --runtime=nvidia \
    --name=ros_melodic \
    ros_melodic \
    bash