import pyshark
import pandas as pd

# Fonction pour extraire un fichier PCAP et le convertir en DataFrame
def extract_pcap_to_dataframe(pcap_file):
    # Capture des paquets dans le fichier PCAP
    cap = pyshark.FileCapture(pcap_file)
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

    # Retourner le DataFrame avec les données extraites
    return pd.DataFrame(data)


# Extraction des deux types de trafic (normal et anormal)
df_normal = extract_pcap_to_dataframe('normal.pcap')  # Fichier PCAP normal
df_anormal = extract_pcap_to_dataframe('anormal.pcap')  # Fichier PCAP anormal

# Ajout de la colonne label pour chaque dataset
df_normal['label'] = 0  # 0 pour normal
df_anormal['label'] = 1  # 1 pour anormal

# Suppression des colonnes inutiles si nécessaire (par exemple 'time' si non pertinente)
cols_to_remove = ['time']
df_normal = df_normal.drop(columns=cols_to_remove, errors='ignore')
df_anormal = df_anormal.drop(columns=cols_to_remove, errors='ignore')

# Fusionner les deux datasets (normal et anormal) pour l'entraînement
df_final = pd.concat([df_normal, df_anormal], ignore_index=True)

# Encodage du protocole (pour le rendre numérique)
protocol_map = {'ICMP': 1, 'TCP': 2, 'UDP': 3, 'UNKNOWN': -1}
df_final['protocol'] = df_final['protocol'].map(protocol_map).fillna(-1)

# Vérifier les statistiques pour avoir un aperçu de l'équilibre des labels
print("Distribution des labels :")
print(df_final['label'].value_counts())

# Sauvegarder le dataset final sous forme de CSV
df_final.to_csv('dataset_final_with_request_reply.csv', index=False)

print("Données traitées et sauvegardées dans 'dataset_final_with_request_reply.csv'.")
