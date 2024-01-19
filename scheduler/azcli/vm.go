package azcli

import (
	"github.com/lgarithm/proc"
)

func Login() proc.Proc {
	return proc.Proc{Prog: `az`, Args: []string{`login`}}
}

func ListResource() proc.Proc {
	return proc.Proc{
		Prog: `az`, Args: []string{`resource`, `list`, `-o`, `table`},
	}
}

func CreateVM(name string, relay, admin string, location, group string, size, image string) proc.Proc {
	return proc.Proc{
		Prog: `az`,
		Args: []string{
			`vm`, `create`,
			`--admin-username`, admin,
			`-l`, location,
			`-g`, group,
			`--nsg`, relay + `NSG`,
			`--vnet-name`, relay + `VNET`,
			`--subnet`, relay + `Subnet`,
			`--public-ip-address`, ``,
			`--size`, size,
			`--image`, image,
			`--os-disk-size-gb`, `100`,
			`-n`, name,
			`--debug`,
			`-o`, `table`,
		},
	}
}

func CreatePublicVM(name string, relay, admin string, location, group string, size, image string) proc.Proc {
	return proc.Proc{
		Prog: `az`,
		Args: []string{
			`vm`, `create`,
			`--admin-username`, admin,
			`-l`, location,
			`-g`, group,
			`--nsg`, relay + `NSG`,
			`--vnet-name`, relay + `VNET`,
			`--subnet`, relay + `Subnet`,
			`--size`, size,
			`--image`, image,
			`--os-disk-size-gb`, `100`,
			`-n`, name,
			`--debug`,
			`-o`, `table`,
		},
	}
}

func StartVM(name string, group string) proc.Proc {
	return proc.Proc{
		Prog: `az`,
		Args: []string{
			`vm`, `start`,
			`-g`, group,
			`-n`, name,
			`-o`, `table`,
		},
	}
}

func StopVM(name string, group string) proc.Proc {
	return proc.Proc{
		Prog: `az`,
		Args: []string{
			`vm`, `deallocate`,
			`-g`, group,
			`-n`, name,
			`-o`, `table`,
		},
	}
}

func DeleteVM(name string, group string) proc.Proc {
	return proc.Proc{
		Prog: `az`,
		Args: []string{
			`vm`, `delete`,
			`-g`, group,
			`-n`, name,
			`--yes`,
			`-o`, `table`,
		},
	}
}

func GetIP(name string, group string) proc.Proc {
	return proc.Proc{
		Prog: `az`,
		Args: []string{
			`vm`, `list-ip-addresses`,
			`-g`, group,
			`-n`, name,
			`--query`, `[0].virtualMachine.network.privateIpAddresses[0]`,
		},
	}
}

func GetPubIP(name string, group string) proc.Proc {
	return proc.Proc{
		Prog: `az`,
		Args: []string{
			`vm`, `list-ip-addresses`,
			`-g`, group,
			`-n`, name,
			`--query`, `[0].virtualMachine.network.publicIpAddresses[0].ipAddress`,
		},
	}
}

func DeleteNIC(name string, group string) proc.Proc {
	return proc.Proc{
		Prog: `az`,
		Args: []string{
			`network`, `nic`,
			`delete`,
			`-g`, group,
			`-n`, name,
			// `--debug`,
		},
	}
}

func DeletePublicIP(name string, group string) proc.Proc {
	return proc.Proc{
		Prog: `az`,
		Args: []string{
			`network`, `public-ip`,
			`delete`,
			`-g`, group,
			`-n`, name,
			// `--debug`,
		},
	}
}

func ListDisk(prefix string, group string) proc.Proc {
	return proc.Proc{
		Prog: `az`,
		Args: []string{
			`disk`, `list`,
			`-g`, group,
			`--query`,
			`[].name | [?starts_with(@, '` + prefix + `')]`,
			// `--debug`,
		},
	}
}

func DeleteDisk(name string, group string) proc.Proc {
	return proc.Proc{
		Prog: `az`,
		Args: []string{
			`disk`, `delete`,
			`-g`, group,
			`-n`, name,
			`--yes`,
			// `--debug`,
		},
	}
}
