package pid

import (
	"encoding/binary"
	"errors"
	"net"
)

type IPv4 uint32

type Port uint16

func FormatIPv4(ipv4 uint32) string {
	ip := net.IPv4(byte(ipv4>>24), byte(ipv4>>16), byte(ipv4>>8), byte(ipv4))
	return ip.String()
}

var (
	errInvalidIPv4 = errors.New("invalid IPv4")
	errInvalidPort = errors.New("invalid port")
)

func ParseIPv4(host string) (uint32, error) {
	ip := net.ParseIP(host)
	if ip == nil {
		return 0, errInvalidIPv4
	}
	ip = ip.To4()
	if ip == nil {
		return 0, errInvalidIPv4
	}
	return PackIPv4(ip), nil
}

func PackIPv4(ip net.IP) uint32 {
	a := uint32(ip[0]) << 24
	b := uint32(ip[1]) << 16
	c := uint32(ip[2]) << 8
	d := uint32(ip[3])
	return a | b | c | d
}

func MustParseIPv4(host string) uint32 {
	ipv4, err := ParseIPv4(host)
	if err != nil {
		panic(err)
	}
	return ipv4
}

var errNoIPv4Found = errors.New("no ipv4 found")

func InferIPv4(nic string) (uint32, error) {
	i, err := net.InterfaceByName(nic)
	if err != nil {
		return 0, err
	}
	addrs, err := i.Addrs()
	if err != nil {
		return 0, errNoIPv4Found
	}
	for _, addr := range addrs {
		var ip net.IP
		switch v := addr.(type) {
		case *net.IPNet:
			ip = v.IP
		case *net.IPAddr:
			ip = v.IP
		}
		if ip != nil {
			ip = ip.To4()
		}
		if ip != nil {
			return PackIPv4(ip), nil
		}
	}

	return 0, errNoIPv4Found
}

func InferBroadcastIPv4(nic string) (uint32, error) {
	i, err := net.InterfaceByName(nic)
	if err != nil {
		return 0, err
	}
	addrs, err := i.Addrs()
	if err != nil {
		return 0, err
	}
	for _, addr := range addrs {
		switch v := addr.(type) {
		case *net.IPNet:
			ip, err := broadcastAddr(v)
			if err != nil {
				return 0, err
			}
			return PackIPv4(ip), nil
		}
	}
	return 0, errNoIPv4Found
}

func broadcastAddr(n *net.IPNet) (net.IP, error) {
	if n.IP.To4() == nil {
		return net.IP{}, errors.New("not IPv4")
	}
	ip := make(net.IP, len(n.IP.To4()))
	binary.BigEndian.PutUint32(ip, binary.BigEndian.Uint32(n.IP.To4())|^binary.BigEndian.Uint32(net.IP(n.Mask).To4()))
	return ip, nil
}
