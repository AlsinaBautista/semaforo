#!/bin/bash
# =============================================================================
# Generación de Certificados para Mutual TLS (mTLS) - Semáforo Inteligente
# =============================================================================

CERTS_DIR="$(dirname "$0")/../configs/certs"
mkdir -p "$CERTS_DIR"
cd "$CERTS_DIR" || exit 1

echo "=== 1. Generando Autoridad Certificante (CA) ==="
# Llave privada de la CA
openssl genrsa -out ca.key 2048
# Certificado raíz de la CA (válido por 10 años)
openssl req -new -x509 -days 3650 -key ca.key -out ca.crt -subj "/C=AR/ST=BuenosAires/L=Necochea/O=SemaforoInteligente/OU=IT/CN=SemaforoCA"

echo "=== 2. Generando Certificado del Servidor (Mosquitto) ==="
# Llave del servidor
openssl genrsa -out server.key 2048
# Certificate Signing Request (CSR)
openssl req -new -key server.key -out server.csr -subj "/C=AR/ST=BuenosAires/L=Necochea/O=SemaforoInteligente/OU=Broker/CN=localhost"
# Firmar el certificado del servidor con la CA
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server.crt -days 3650 -sha256

echo "=== 3. Generando Certificado Cliente: Coordinator ==="
openssl genrsa -out coordinator.key 2048
openssl req -new -key coordinator.key -out coordinator.csr -subj "/C=AR/ST=BuenosAires/L=Necochea/O=SemaforoInteligente/OU=Brain/CN=coordinator"
openssl x509 -req -in coordinator.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out coordinator.crt -days 3650 -sha256

echo "=== 4. Generando Certificado Cliente: Edge-001 ==="
openssl genrsa -out edge-001.key 2048
# El Common Name (CN) es CRUCIAL. Mosquitto usa use_identity_as_username, así que CN=edge-001 mapeará a las reglas ACL.
openssl req -new -key edge-001.key -out edge-001.csr -subj "/C=AR/ST=BuenosAires/L=Necochea/O=SemaforoInteligente/OU=Edge/CN=edge-001"
openssl x509 -req -in edge-001.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out edge-001.crt -days 3650 -sha256

echo "Limpiando archivos CSR temporales..."
rm *.csr
rm ca.srl

echo "Certificados generados exitosamente en $CERTS_DIR:"
ls -l
