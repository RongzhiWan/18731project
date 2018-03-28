import os
import glob
import numpy as np
import dpkt
import json
import socket

# Input:    31 folders, each contains number of ".pcap" files
# Output:   27 ".txt" files, each corresponds to some meta data and a flatten 23*N matrix
#           1 README.txt, contains typeName<->number mapping info

name_mapper = {}    # typeName<->number mapper
device_num = 0      # initial value for type
type_mapper = {}
type_num = 0
ip_counter = {}     # distinct ip Addr counter
ip_cnt = 1          # initial value for dst IP addr counter


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
for type_path in glob.glob(os.path.join(path, '*/')):           # for each type folder
    type_str = type_path.split("/")[3]
    print(type_path)
    name_mapper[type_str] = device_num  # map type name to type number
    doIncre = True
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
    F = np.zeros((features, 0)).astype(int)

    for cap in glob.glob(os.path.join(type_path, '*.pcap')):        # for each capture

        for ts, pkt in dpkt.pcap.Reader(open(cap, 'rb')):           # for each packet
            col = np.zeros((features, 1)).astype(int)   # initialize a column vector for this packet
            eth = dpkt.ethernet.Ethernet(pkt)
            if eth.type == dpkt.ethernet.ETH_TYPE_ARP:      # f1: ARP
                col[0, 0] = 1
                continue    # because ARP doesn't have upper stacks

            if isinstance(eth.data, dpkt.llc.LLC):          # f2: LLC
                col[1, 0] = 1
                continue

            if eth.type == dpkt.ethernet.ETH_TYPE_IP6:
                ipv6 = eth.data
                if isinstance(ipv6.data, dpkt.icmp.ICMP):   # f5: ICMPv6
                    col[4, 0] = 1
                continue

            # ref: www.networksorcery.com/enp/protocol/802/ethertypes.htm
            # not sure if this works, because dpkt doesn't have built-in EAPoL type
            if eth.type == 0x888E:                          # f6: EAPoL
                col[5, 0] = 1
                continue

            if eth.type == dpkt.ethernet.ETH_TYPE_IP:       # f3: IP
                col[2, 0] = 1
                ip = eth.data
                col[18, 0] = ip.len                         # f19: Packet size
                if len(ip.data) > 0:
                    col[19, 0] = 1                          # f20: raw data
                dstIP = socket.inet_ntoa(ip.dst)
                if dstIP in ip_counter:   # seen this IP before
                    col[20, 0] = ip_counter[dstIP]
                else:                     # not seen before
                    ip_cnt += 1
                    ip_counter[dstIP] = ip_cnt
                    col[20, 0] = ip_cnt                     # f21: dst IP addr counter

                if ip.opts.find(b'\x94\x04') != -1:
                    # print(ip.opts)
                    # print(cap)
                    col[17, 0] = 1                          # f18: Router Alert
                else:
                    col[16, 0] = 1

                if isinstance(ip.data, dpkt.icmp.ICMP):     # f4: ICMP
                    col[3, 0] = 1
                    continue

                try:
                    col[21, 0] = port_mapper(ip.data.sport) # f22: Source port
                except AttributeError:  # no port
                    col[21, 0] = 0

                try:
                    col[22, 0] = port_mapper(ip.data.dport) # f23: Destination port
                except AttributeError:
                    col[22, 0] = 0

                if ip.p == dpkt.ip.IP_PROTO_TCP:            # f7: TCP
                    col[6, 0] = 1
                    tcp = ip.data
                    if tcp.dport == 80:                     # f9: HTTP
                        col[8, 0] = 1
                    elif tcp.dport == 443:                  # f10: HTTPS
                        col[9, 0] = 1
                    # ref: support.microsoft.com/en-us/help/832017
                    elif tcp.dport == 2869 or tcp.dport == 5000:
                        col[12, 0] = 1                      # f13: SSDP

                if ip.p == dpkt.ip.IP_PROTO_UDP:            # f8: UDP
                    col[7, 0] = 1
                    udp = ip.data
                    # BOOTP is used by DHCP and some other protocols
                    if udp.dport == 67 or udp.dport == 68:
                        col[11, 0] = 1                      # f12: BOOTP
                        if isinstance(udp.data, dpkt.dhcp.DHCP):
                            col[10, 0] = 1                  # f11: DHCP
                    elif udp.dport == 1900:
                        col[12, 0] = 1                      # f13: SSDP
                    elif udp.dport == 53:
                        col[13, 0] = 1                      # f14: DNS
                    elif udp.dport == 5353:
                        col[14, 0] = 1                      # f15: mDNS
                    elif udp.dport == 123:
                        col[15, 0] = 1                      # f16: NTP

            # add this column vector to F
            F = np.hstack((F, col))
            # end packet loop

    # convert to training data format
    F_str = flatten(F)

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
    result = str(in_matrix_type_num) + '\t' + str(F.shape[0]) + '\t' + str(F.shape[1]) + '\t' + F_str

    print(F.shape)
    print('after split')
    print(len(F_str.split(' ')))

    # save training data to a text file
    with open('../data/output/' + str(device_num) + '.txt', "w") as file:
        file.write("%s" % result)
        file.close()

    # increment type number
    device_num += 1
    if doIncre:
        type_num += 1

print(name_mapper)
# save typeName<->number conversion to text file
with open('../data/output/README.txt', 'w') as readMe:
    readMe.write(json.dumps(name_mapper))