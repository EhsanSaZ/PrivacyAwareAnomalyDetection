{0: 0, 1: 1, 4: 2, 7: 3, 10: 4, 13: 5, 33: 6, 37: 7, 43: 8}

0: 0 - Normal transfer
1: 1 - Read congestion on the sender storage (OST_read_congestion_by_clients_sender_ost)
4: 2 - Write congestion on the recevier storage (write_congestion_by_clients_on_receiver_ost_3)
7: 3 - Netwrok Congestion when reading from storage (Lustre_Link_congestion_on_sender_side)
10: 4 - Netwrok Congestion when writing to receiver storage (Lustre_Link_congestion_on_receiver_side)
13: 5 - Netwrok packet loss anomaly (network_anomaly_network_loss)
33: 6 - Netwrok congestion between two hosts (network_anomaly_network_congestion)
37: 7 - Config anomaly TCP send buffer setting on sender side (sys_config_tcp_send)
43: 8 - Config anomaly TCP receive buffer setting on receiver side (sys_config_tcp_receive_buffer)

17: network_anomaly_network_jitter
21: network_anomaly_network_duplicate
25: network_anomaly_network_corrupt
29: network_anomaly_network_reorder