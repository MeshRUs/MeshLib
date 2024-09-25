#!/bin/bash

if command -v udevadm 2>&1 >/dev/null ; then
  echo "Updating udev rules"
  udevadm control --reload-rules && udevadm trigger
fi

echo "Updating ldconfig"
cat <<EOF > /etc/ld.so.conf.d/meshlib_libs.conf
/usr/local/lib/MeshLib
/usr/local/lib/MeshLib/lib
EOF
ldconfig
