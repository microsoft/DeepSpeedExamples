#!/bin/bash
hostfile=/job/hostfile
PIP_SUDO="sudo -H"
PIP_INSTALL="pip install -r"
PIP_SHOW="pip show"
PKG="qanet"
REQ_FILE="requirements.txt"

tmp_wheel_path="/tmp/${PKG}"
hosts=`cat $hostfile | awk '{print $1}' | paste -sd "," -`;
pkgs=`cat ${REQ_FILE} | awk 'BEGIN { FS =  "=="}; {print $1}' | paste -sd " " -`;
export PDSH_RCMD_TYPE=ssh;

pdsh -w $hosts "if [ -d $tmp_wheel_path ]; then rm -f $tmp_wheel_path/*; else mkdir -pv $tmp_wheel_path; fi"
pdcp -w $hosts ${REQ_FILE} ${tmp_wheel_path}/
pdsh -w ${hosts} "${PIP_SUDO} ${PIP_INSTALL} ${tmp_wheel_path}/${REQ_FILE}"
pdsh -w ${hosts} "${PIP_SUDO} ${PIP_SHOW} ${pkgs}"
echo "${PKG} Installation is successful"
pdsh -w $hosts "if [ -d $tmp_wheel_path ]; then rm -rf $tmp_wheel_path/ ; fi"