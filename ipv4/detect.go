package ipv4

import "net"

func Detect(nicName string) string {
	nics, err := net.Interfaces()
	if err != nil {
		return ""
	}
	for _, nic := range nics {
		if len(nicName) > 0 && nicName != nic.Name {
			continue
		}
		addrs, err := nic.Addrs()
		if err != nil {
			continue
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
				// fmt.Printf("%s %s\n", nic.Name, ip.String())
				return ip.String()
			}
		}
	}
	return ""
}
