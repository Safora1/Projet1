import pyshark
import pandas as pd

# Fonction pour extraire un fichier PCAP et le convertir en DataFrame
def extract_pcap_to_dataframe(pcap_file):
    # Capture des paquets dans le fichier PCAP
    cap = pyshark.FileCapture(pcap_file, use_json=True)  # Utilisation de json pour éviter les problèmes de boucle asynchrone
    data = []

    # Itérer à travers chaque paquet dans le fichier PCAP
    for pkt in cap:
        try:
            # Vérifier que le paquet contient un en-tête IP
            if 'IP' in pkt:
                # Extraire le protocole (ICMP, TCP, UDP, etc.)
                protocol = pkt.transport_layer if hasattr(pkt, 'transport_layer') else 'UNKNOWN'

                # Déterminer si c'est une requête ou une réponse (basé sur ICMP)
                if 'ICMP' in pkt:
                    if hasattr(pkt.icmp, 'type'):
                        if pkt.icmp.type == '8':  # Type 8 est pour les requêtes Echo (ping request)
                            request_reply = 'request'
                        elif pkt.icmp.type == '0':  # Type 0 est pour les réponses Echo (ping reply)
                            request_reply = 'reply'
                        else:
                            request_reply = 'unknown'
                    else:
                        request_reply = 'unknown'
                else:
                    request_reply = 'unknown'

                # Créer un dictionnaire pour chaque paquet avec ses caractéristiques
                row = {
                    'src_ip': pkt.ip.src,  # Adresse IP source
                    'dst_ip': pkt.ip.dst,  # Adresse IP destination
                    'protocol': protocol,  # Protocole (ICMP, TCP, UDP)
                    'length': int(pkt.length),  # Longueur du paquet
                    'time': str(pkt.sniff_time),  # Heure du paquet capturé
                    'ttl': int(pkt.ip.ttl) if hasattr(pkt.ip, 'ttl') else None,  # TTL
                    'id': int(pkt.icmp.id) if 'ICMP' in pkt else None,  # ID du paquet ICMP (si ICMP)
                    'seq': int(pkt.icmp.seq) if 'ICMP' in pkt else None,  # Séquence du paquet ICMP (si ICMP)
                    'request_reply': request_reply  # 'request' ou 'reply'
                }
                data.append(row)
        except AttributeError:
            continue  # Ignorer les paquets sans les attributs nécessaires

    # Retourner un DataFrame avec les données extraites
    return pd.DataFrame(data)


# Extraction des données depuis les fichiers PCAP
df_normal = extract_pcap_to_dataframe('normal.pcap')  # Fichier PCAP normal
df_anormal = extract_pcap_to_dataframe('anomal.pcap')  # Fichier PCAP anormal

# Ajouter une colonne 'label' pour chaque type de trafic (0 pour normal, 1 pour anormal)
df_normal['label'] = 0  # 0 pour normal
df_anormal['label'] = 1  # 1 pour anormal

# Fusionner les deux datasets
df_final = pd.concat([df_normal, df_anormal], ignore_index=True)

# Sauvegarder le DataFrame final sous forme de fichier CSV
df_final.to_csv('dataset_cleaned.csv', index=False)

print("Données extraites et sauvegardées dans 'dataset_cleaned.csv'.")
