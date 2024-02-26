package experiments

import "github.com/lgarithm/proc/experimental"

func ReInstallMLFS(a At) P {
	const script = `
set -e
echo "deb https://europe-west2-apt.pkg.dev/projects/tenplex tenplex main" | sudo tee /etc/apt/sources.list.d/tenplex.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/packages-cloud-google-apt.gpg >/dev/null
sudo apt update
sudo apt remove -y mlfs
sudo apt reinstall -y mlfs
sudo systemctl stop mlfs
sudo systemctl start mlfs
mlfs info
`
	return seq(
		runScript(a, script, `install-mlfs.sh`, false),
	)
}

func runScript(a At, script string, filename string, su bool) P {
	run := a.PC(`./` + filename)
	if su {
		run = a.PC(`sudo `, `./`+filename)
	}
	return seq(
		touchExe(a, filename, []byte(script)),
		run,
	)
}

var touchExe = experimental.TouchExe
