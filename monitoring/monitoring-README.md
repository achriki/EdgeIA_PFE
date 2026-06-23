# T21 — Prometheus + Grafana (Docker Compose sur PC)

## Structure

```
monitoring/
├── docker-compose.yml                          # stack Prometheus + Grafana
├── prometheus.yml                              # config scrape (à adapter avec l'IP du Pi)
├── grafana/
│   ├── provisioning/
│   │   ├── datasources/prometheus.yml         # datasource auto-provisionnée
│   │   └── dashboards/default.yml             # chemin des dashboards
│   └── dashboards/
│       └── edgeia-dashboard.json              # dashboard (4 panneaux)
└── README.md
```

## Étapes

### 1. Trouver l'IP du Raspberry Pi

```bash
# Sur le Pi
hostname -I
# ex : 192.168.1.42
```

### 2. Remplacer PI_IP dans prometheus.yml

```bash
# Sur le PC, dans le dossier monitoring/
sed -i 's/PI_IP/192.168.1.42/g' prometheus.yml   # Linux/Mac
# ou éditez prometheus.yml manuellement (cherchez PI_IP)
```

### 3. Démarrer la stack

```bash
cd monitoring/
docker compose up -d
docker compose ps   # vérifier que les 2 containers sont Up
```

### 4. Vérifier les cibles Prometheus

Ouvrir : http://localhost:9090/targets

Vous devez voir 3 cibles :
| Job | Target | Statut attendu |
|---|---|---|
| prometheus | localhost:9090 | UP |
| node-exporter | PI_IP:9100 | UP |
| inference-app | PI_IP:30080 | UP* |

> *`inference-app` sera DOWN si le pod K3s n'est pas en cours d'exécution.
> Lancez une inférence test : `curl http://PI_IP:30080/health`

### 5. Accéder à Grafana

URL : http://localhost:3000  
Login : `admin` / `admin` (changer au 1er login)

Le dashboard **"EdgeIA - Monitoring Pi 3B"** est automatiquement disponible via le provisioning.

## Rechargement à chaud de la config Prometheus

Si vous modifiez `prometheus.yml` :

```bash
curl -X POST http://localhost:9090/-/reload
```

## Arrêt de la stack

```bash
docker compose down          # arrêt + suppression des containers
docker compose down -v       # + suppression des volumes (données perdues)
```

## Métriques exposées par l'inference-app (T19)

| Métrique | Type | Description |
|---|---|---|
| `inference_requests_total` | Counter | Nombre total de requêtes /predict |
| `persons_detected_total` | Counter | Nombre total de personnes détectées |
| `inference_latency_seconds` | Histogram | Latence d'inférence (buckets : 50ms→2s) |

## Métriques Node Exporter utilisées (T20)

| Métrique | Panel |
|---|---|
| `node_cpu_seconds_total` | CPU % |
| `node_memory_MemAvailable_bytes` | RAM % |
| `node_thermal_zone_temp` | Température SoC |
