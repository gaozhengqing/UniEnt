#!/bin/bash

set -Eeuxo pipefail

while getopts ":p:" opt; do
    case "$opt" in
        p)
            path="$OPTARG"
            ;;
    esac
done

cd $path

# download
wget --content-disposition https://zenodo.org/records/2235448/files/blur.tar?download=1
wget --content-disposition https://zenodo.org/records/2235448/files/digital.tar?download=1
wget --content-disposition https://zenodo.org/records/2235448/files/extra.tar?download=1
wget --content-disposition https://zenodo.org/records/2235448/files/noise.tar?download=1
wget --content-disposition https://zenodo.org/records/2235448/files/weather.tar?download=1

# extract
tar -xvf blur.tar
tar -xvf digital.tar
tar -xvf extra.tar
tar -xvf noise.tar
tar -xvf weather.tar
