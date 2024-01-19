package tfrecord

import (
	"bytes"
	"encoding/binary"
	"errors"
	"hash/crc32"
	"io"
)

// https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/load_data/tfrecord.ipynb?hl=sr&skip_cache=true#scrollTo=o6qxofy89obI
type TFRecordHead struct {
	Len int64

	// masked_crc = ((crc >> 15) | (crc << 17)) + 0xa282ead8ul
	LenCRC  uint32 // masked_crc32_of_length
	DataCRC uint32 // masked_crc32_of_data
}

func MaskedCRC(bs []byte) uint32 {
	return maskCRC(crc32sum(bs))
}

func maskCRC(crc uint32) uint32 {
	return ((crc >> 15) | (crc << 17)) + uint32(0xa282ead8)
}

func crc32sum(bs []byte) uint32 {
	table := crc32.MakeTable(crc32.Castagnoli)
	return crc32.Checksum(bs, table)
}

var errinvalidDataCrc = errors.New("invalid data crc")

var endian = binary.LittleEndian

func ReadTFRecord(f io.Reader) (*TFRecordHead, []byte, error) {
	var info TFRecordHead
	if err := binary.Read(f, endian, &info.Len); err != nil {
		return nil, nil, err
	}
	if err := binary.Read(f, endian, &info.LenCRC); err != nil {
		return nil, nil, unexpectEOF(err)
	}
	bs := make([]byte, int(info.Len))
	if _, err := io.ReadFull(f, bs); err != nil {
		return nil, nil, unexpectEOF(err)
	}
	if err := binary.Read(f, endian, &info.DataCRC); err != nil {
		return nil, nil, unexpectEOF(err)
	}
	if info.DataCRC != MaskedCRC(bs) {
		return nil, nil, errinvalidDataCrc
	}
	return &info, bs, nil
}

func WriteTFRecord(bs []byte, f io.Writer) error {
	info := TFRecordHead{
		Len:     int64(len(bs)),
		DataCRC: MaskedCRC(bs),
	}
	buf := &bytes.Buffer{}
	if err := binary.Write(buf, endian, &info.Len); err != nil {
		return err
	}
	info.LenCRC = MaskedCRC(buf.Bytes())
	if err := binary.Write(f, endian, &info.Len); err != nil {
		return err
	}
	if err := binary.Write(f, endian, &info.LenCRC); err != nil {
		return err
	}
	if _, err := f.Write(bs); err != nil {
		return err
	}
	if err := binary.Write(f, endian, &info.DataCRC); err != nil {
		return err
	}
	return nil
}

func unexpectEOF(err error) error {
	if err == io.EOF {
		return io.ErrUnexpectedEOF
	}
	return err
}
