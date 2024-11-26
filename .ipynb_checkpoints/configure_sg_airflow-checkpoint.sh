#!/bin/bash

# Array de IDs dos grupos de segurança
SECURITY_GROUP_IDS=("sg-021730aeec07d54c8" "sg-04e18a05f58de3a43" "sg-e9bcdbd3" "sg-009d9f9acada6eb2c" "sg-04f90bd4580cf3083" "sg-0624795c05fd117ac" "sg-06c2f333cf60f9549" "sg-0b931ce9e78a019d3" "sg-0f9b696fc319754ef" "sg-0b71893139388d3c5" "sg-0ec9a63a99cef388e" "sg-09e6960525fd0062e" "sg-047bb1e7593c93d57" "sg-0dd8d5ae9ca447fdc" "sg-0f81f9e6ab865c2b9" "sg-03867c9f6158c1047" "sg-03f802089c78cc7b7" "sg-0e2c7275c335f20f3" "sg-0da1df081017fe390" "sg-066bb02681480992a" "sg-049b787b284719b05" "sg-0a4bf83755ba1730f" "sg-0b1996a6b7c3e3fc8" "sg-02ade590922a2e238" "sg-0951f4f880c58b074" "sg-07c29c3c97f834e4e" "sg-0690517b050d21386" "sg-09467f76417f33aa1" "sg-07436bfb83fe902d5" "sg-0801a942c4e06004b" "sg-050fe8ab2b76b93a1" "sg-08380b6428808f0b6" "sg-0857efef62c876c50" "sg-05d4a4e0db36dd77c" "sg-0fff9a188c4a42382" "sg-06af6b4e8d9d8f40b" "sg-01c16ecb21a061fed" "sg-00951fbcbe8926f91" "sg-0dd2b9012943cd466" "sg-0fc19ef7805f29aeb" "sg-08b83f5333601c1c6" "sg-0b4940bc8d0778542")

# IP que terá acesso ao HTTP (porta 8080)
MY_IP="0.0.0.0/0"

# IP do serviço EC2 Instance Connect
INSTANCE_CONNECT_IP="18.206.107.24/29"

# Iterar sobre cada grupo de segurança e adicionar as regras de entrada
for SG_ID in "${SECURITY_GROUP_IDS[@]}"
do
  # Adiciona regra para a porta 8080 (HTTP)
  aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port 8080 --cidr $MY_IP
  
  # Adiciona regra para a porta 22 (SSH) para o serviço EC2 Instance Connect
  aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port 22 --cidr $INSTANCE_CONNECT_IP
done
