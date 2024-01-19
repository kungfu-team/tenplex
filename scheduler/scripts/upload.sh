#!/bin/sh
set -e

if [ -z "$SAS" ]; then
    echo "SAS NOT set"
    exit 1
fi

upload() {
    local filename=$1
    local path=$2
    SA=tenplex
    URI="https://$SA.blob.core.windows.net/$path"

    echo "uploading $filename to $URI"
    curl -s -w "\n%{http_code}\n" -X PUT \
        -H 'x-ms-blob-type: BlockBlob' \
        -H 'x-ms-version: 2015-02-21' \
        -H "Content-Type: $ContentType" \
        "$URI?$SAS" --data-binary @$filename
    echo "uploaded $URI"
}

upload "$1" "$2"
