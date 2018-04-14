import glob
import os
import socket

import dpkt
import numpy as np

''' 
INPUT
31 folders, each contains number of ".pcap" files

OUTPUT
27 ".txt" files, each corresponds to multiple observations of one type

FEATURES
1.ARP  2.LLC
3.IP  4.ICMP  5.ICMPv6  6.EAPoL
7.TCP  8.UDP
9.HTTP  10.HTTPS  11.DHCP  12.BOOTP  13.SSDP  14.DNS  15.MDNS  16.NTP  17.Padding  18.RouterAlert
19.Size (int)  20.Raw data
21.Destination IP counter (int)
22.Source (int)  23.Destination (int)
'''

name_mapper = {}  # typeName<->number mapper
device_num = 0  # initial value for type
type_mapper = {}
type_num = 0
ip_counter = {}  # distinct ip Addr counter
ip_cnt = 1  # initial value for dst IP addr counter


def port_mapper(port):
    if port in range(0, 1023):
        return 1
    elif port in range(1024, 49151):
        return 2
    elif port in range(49152, 65535):
        return 3


def flatten(matrix):
    row, column = matrix.shape
    n = row * column
    one_d_array = np.reshape(matrix, n)

    list = []
    for e in one_d_array:
        list.append(str(e))

    return " ".join(list)


path = '../data/capture'
for type_path in glob.glob(os.path.join(path, '*/')):  # for each device
    type_str = type_path.split("/")[3]
    name_mapper[type_str] = device_num  # map type name to type number
    doIncre = True
    print(type_str)
    if type_str == 'EdimaxCam1' or type_str == 'EdimaxCam2':
        if 'EdimaxCam' not in type_mapper:
            type_mapper['EdimaxCam'] = type_num
        else:
            doIncre = False
    elif type_str == 'EdnetCam1' or type_str == 'EdnetCam2':
        if 'EdnetCam' not in type_mapper:
            type_mapper['EdnetCam'] = type_num
        else:
            doIncre = False
    elif type_str == 'WeMoInsightSwitch' or type_str == 'WeMoInsightSwitch2':
        if 'WeMoInsightSwitch' not in type_mapper:
            type_mapper['WeMoInsightSwitch'] = type_num
        else:
            doIncre = False
    elif type_str == 'WeMoSwitch' or type_str == 'WeMoSwitch2':
        if 'WeMoSwitch' not in type_mapper:
            type_mapper['WeMoSwitch'] = type_num
        else:
            doIncre = False
    else:
        type_mapper[type_str] = type_num

    features = 23
    F = ''

    for cap in glob.glob(os.path.join(type_path, '*.pcap')):  # for each capture
        F_cap = np.zeros((features, 0)).astype(int)
        for ts, pkt in dpkt.pcap.Reader(open(cap, 'rb')):  # for each packet
            col = np.zeros((features, 1)).astype(int)  # initialize a column vector for this packet
            eth = dpkt.ethernet.Ethernet(pkt)
            if eth.type == dpkt.ethernet.ETH_TYPE_ARP:  # f1: ARP
                col[0, 0] = 1
                continue

            if isinstance(eth.data, dpkt.llc.LLC):  # f2: LLC
                col[1, 0] = 1

            if eth.type == dpkt.ethernet.ETH_TYPE_IP6:
                ipv6 = eth.data
                if isinstance(ipv6.data, dpkt.icmp.ICMP):  # f5: ICMPv6
                    col[4, 0] = 1
                continue

            # ref: www.networksorcery.com/enp/protocol/802/ethertypes.htm
            # not sure if this works, because dpkt doesn't have built-in EAPoL type
            if eth.type == 0x888E:  # f6: EAPoL
                col[5, 0] = 1
                continue

            # "or eth.type == dpkt.ethernet.ETH_TYPE_IP6" ?
            if eth.type == dpkt.ethernet.ETH_TYPE_IP:
                col[2, 0] = 1  # f3: IP
                ip = eth.data
                col[18, 0] = ip.len  # f19: Packet size
                if len(ip.data) > 0:
                    col[19, 0] = 1  # f20: raw data
                dstIP = socket.inet_ntoa(ip.dst)
                if dstIP in ip_counter:  # seen this IP before
                    col[20, 0] = ip_counter[dstIP]
                else:  # not seen before
                    ip_cnt += 1
                    ip_counter[dstIP] = ip_cnt
                    col[20, 0] = ip_cnt  # f21: dst IP addr counter

                if ip.opts.find(b'\x94\x04') != -1:
                    # print(ip.opts)
                    # print(cap)
                    col[17, 0] = 1  # f18: Router Alert
                else:
                    col[16, 0] = 1

                if isinstance(ip.data, dpkt.icmp.ICMP):  # f4: ICMP
                    col[3, 0] = 1
                    continue

                try:
                    col[21, 0] = port_mapper(ip.data.sport)  # f22: Source port
                except AttributeError:  # no port
                    col[21, 0] = 0

                try:
                    col[22, 0] = port_mapper(ip.data.dport)  # f23: Destination port
                except AttributeError:
                    col[22, 0] = 0

                if ip.p == dpkt.ip.IP_PROTO_TCP:  # f7: TCP
                    col[6, 0] = 1
                    tcp_port = ip.data.dport
                    if tcp_port == 80:  # f9: HTTP
                        col[8, 0] = 1
                    elif tcp_port == 443:  # f10: HTTPS
                        col[9, 0] = 1
                    # ref: support.microsoft.com/en-us/help/832017
                    elif tcp_port == 2869:
                        col[12, 0] = 1  # f13: SSDP

                if ip.p == dpkt.ip.IP_PROTO_UDP:  # f8: UDP
                    col[7, 0] = 1
                    udp_port = ip.data.dport
                    # BOOTP is used by DHCP and some other protocols
                    if udp_port == 67 or udp_port == 68:
                        col[11, 0] = 1  # f12: BOOTP
                        if isinstance(udp_port, dpkt.dhcp.DHCP):
                            col[10, 0] = 1  # f11: DHCP
                    # ref: en.wikipedia.org/wiki/List_of_TCP_and_UDP_port_numbers#cite_note-stackoverflow-323351-42
                    elif udp_port == 1900:
                        col[12, 0] = 1  # f13: SSDP
                    elif udp_port == 5353:
                        col[14, 0] = 1  # f15: mDNS
                    elif udp_port == 123:
                        col[15, 0] = 1  # f16: NTP

                # Protocols over both TCP and UDP
                try:
                    trans_port = ip.data.dport
                    if trans_port == 53:
                        col[13, 0] = 1  # f14: DNS
                except AttributeError:
                    continue

            # for each packet
            F_cap = np.hstack((F_cap, col))

        # for each capture
        F_str = flatten(F_cap)
        if type_str == 'EdimaxCam1' or type_str == 'EdimaxCam2':
            in_matrix_type_num = type_mapper['EdimaxCam']

        elif type_str == 'EdnetCam1' or type_str == 'EdnetCam2':
            in_matrix_type_num = type_mapper['EdnetCam']

        elif type_str == 'WeMoInsightSwitch' or type_str == 'WeMoInsightSwitch2':
            in_matrix_type_num = type_mapper['WeMoInsightSwitch']

        elif type_str == 'WeMoSwitch' or type_str == 'WeMoSwitch2':
            in_matrix_type_num = type_mapper['WeMoSwitch']

        else:
            in_matrix_type_num = type_mapper[type_str]

        result = str(in_matrix_type_num) + '\t' + str(F_cap.shape[0]) + '\t' + str(F_cap.shape[1]) + '\t' + F_str

        # add this capture as a new row
        F = F + result + '\n'

    # for each device
    with open('../data/output/v1/' + str(device_num) + '.txt', "w") as file:
        file.write("%s" % F)
        file.close()

    # increment type number
    device_num += 1
    if doIncre:
        type_num += 1

# save typeName<->number conversion to text file
max_len = max([len(k) for k in name_mapper.keys()])
padding = 4
with open('../data/output/v1/README.txt', 'w') as readMe:
    readMe.write('{k:{k_len:d}s} {v:s}'.format(k_len=max_len + padding, k='Device', v='File number') + '\n')
    for k, v in sorted(name_mapper.items(), key=lambda i: i[1]):
        readMe.write('{k:{k_len:d}s} {v:d}'.format(k_len=max_len + padding, v=v, k=k) + '\n')
