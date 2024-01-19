package pid

import (
	"fmt"
	"net"
	"strconv"
)

// PeerID is the unique identifier of a peer.
type PeerID struct {
	IPv4 uint32
	Port uint16
}

func (p PeerID) String() string {
	return net.JoinHostPort(FormatIPv4(p.IPv4), strconv.Itoa(int(p.Port)))
}

func (p PeerID) ColocatedWith(q PeerID) bool {
	return p.IPv4 == q.IPv4
}

func (p PeerID) ListenAddr(strict bool) PeerID {
	if strict {
		return PeerID{IPv4: p.IPv4, Port: p.Port}
	}
	return PeerID{IPv4: 0, Port: p.Port}
}

func (p PeerID) SockFile() string {
	return fmt.Sprintf(`/tmp/goml-peer-%d.sock`, p.Port)
}

func ParsePeerID(val string) (*PeerID, error) {
	host, p, err := net.SplitHostPort(val)
	if err != nil {
		return nil, err
	}
	ipv4, err := ParseIPv4(host) // FIXME: checkout error
	if err != nil {
		return nil, err
	}
	port, err := strconv.Atoi(p)
	if err != nil {
		return nil, err
	}
	if int(uint16(port)) != port {
		return nil, errInvalidPort
	}
	return &PeerID{
		IPv4: ipv4,
		Port: uint16(port),
	}, nil
}
