resource "digitalocean_droplet" "g2-terraform-droplet" {
  image = "ubuntu-25-04-x64"
  name = "g2-terraform-droplet"
  region = "nyc2"
  size = "s-1vcpu-1gb"
  ssh_keys = [
    data.digitalocean_ssh_key.terraform.id
  ]
  connection {
    host = self.ipv4_address
    user = "root"
    type = "ssh"
    private_key = file(var.pvt_key)
    timeout = "2m"
  }
  provisioner "remote-exec" {
    inline = [
      "export PATH=$PATH:/usr/bin",
      "sleep 30",
      "sudo apt-get update",
      "sudo apt-get -y install nginx"
    ]
  }
}