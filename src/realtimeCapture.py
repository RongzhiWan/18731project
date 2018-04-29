import pcap
import socket

import dpkt
import numpy as np
from dpkt.compat import compat_ord

ip_counter = {}  # distinct ip Addr counter
ip_cnt = 1  # initial value for dst IP addr counter

mac_counter = {}  # distinct mac Addr
mac_cnt = 0  # how many packets have been captured for this MAC

mac_capture = {}  # each mac addr has one mac_capture to save fingerprint


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


def mac_addr(address):
    """Convert a MAC address to a readable/printable string

       Args:
           address (str): a MAC address in hex form (e.g. '\x01\x02\x03\x04\x05\x06')
       Returns:
           str: Printable/readable MAC address
    """
    return ':'.join('%02x' % compat_ord(b) for b in address)


def add_packet_to_fingerprint(srcMAC, dstMAC, mac_counter, mac_capture, finger_num, col):
    # add to srcMAC fingerprint
    counter_temp = mac_counter[srcMAC]
    F_cap_temp = mac_capture[srcMAC]
    if counter_temp < finger_num:
        mac_capture[srcMAC] = np.hstack((F_cap_temp, col))
        mac_counter[srcMAC] += 1
    elif counter_temp == finger_num:
        F_str_temp = flatten(F_cap_temp)
        result_temp = '-1' + '\t' + str(F_cap_temp.shape[0]) + '\t' + str(F_cap_temp.shape[1]) + '\t' + F_str_temp
        F = result_temp + '\n'
        with open('../data/output/' + str(srcMAC) + '.txt', "w") as file:
            file.write("%s" % F)
            file.close()
            # TODO: Classify this MAC

    # add to dstMAC fingerprint
    counter_temp = mac_counter[dstMAC]
    F_cap_temp = mac_capture[dstMAC]
    if counter_temp < finger_num:
        mac_capture[dstMAC] = np.hstack((F_cap_temp, col))
        mac_counter[dstMAC] += 1
    elif counter_temp == finger_num:
        F_str_temp = flatten(F_cap_temp)
        result_temp = '-1' + '\t' + str(F_cap_temp.shape[0]) + '\t' + str(F_cap_temp.shape[1]) + '\t' + F_str_temp
        F = result_temp + '\n'
        with open('../data/output/' + str(dstMAC) + '.txt', "w") as file:
            file.write("%s" % F)
            file.close()
            # TODO: Classify this MAC


sniffer = pcap.pcap(name=None, promisc=True, immediate=True)

features = 28
finger_num = 15

for timestamp, raw_buf in sniffer:
    eth = dpkt.ethernet.Ethernet(raw_buf)

    # check MAC addr
    dstMAC = mac_addr(eth.dst)
    srcMAC = mac_addr(eth.src)

    if srcMAC not in mac_counter:
        print('src MAC: ', srcMAC, ' | dst MAC: ', dstMAC)
        print("Not seen this src MAC addr before")
        mac_counter[srcMAC] = mac_cnt
        F_cap = np.zeros((features, 0)).astype(int)
        mac_capture[srcMAC] = F_cap

    if dstMAC not in mac_counter:
        print('src MAC: ', srcMAC, ' | dst MAC: ', dstMAC)
        print("Not seen this dst MAC addr before")
        mac_counter[dstMAC] = mac_cnt
        F_cap = np.zeros((features, 0)).astype(int)
        mac_capture[dstMAC] = F_cap

    # process each packet
    col = np.zeros((features, 1)).astype(int)  # initialize a column vector for this packet
    col[20, 0] = ip_cnt  # f21: dst IP addr counter

    if eth.type == dpkt.ethernet.ETH_TYPE_ARP:  # f1: ARP
        col[0, 0] = 1
        add_packet_to_fingerprint(srcMAC, dstMAC, mac_counter, mac_capture, finger_num, col)
        continue

    if isinstance(eth.data, dpkt.llc.LLC):  # f2: LLC
        col[1, 0] = 1

    if eth.type == dpkt.ethernet.ETH_TYPE_IP6:
        ipv6 = eth.data
        if isinstance(ipv6.data, dpkt.icmp.ICMP):  # f5: ICMPv6
            col[4, 0] = 1
        add_packet_to_fingerprint(srcMAC, dstMAC, mac_counter, mac_capture, finger_num, col)
        continue

    # ref: www.networksorcery.com/enp/protocol/802/ethertypes.htm
    # not sure if this works, because dpkt doesn't have built-in EAPoL type
    if eth.type == 0x888E:  # f6: EAPoL
        col[5, 0] = 1
        add_packet_to_fingerprint(srcMAC, dstMAC, mac_counter, mac_capture, finger_num, col)
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
            add_packet_to_fingerprint(srcMAC, dstMAC, mac_counter, mac_capture, finger_num, col)
            continue

        if isinstance(ip.data, dpkt.igmp.IGMP):  # f25: IGMP
            col[24, 0] = 1
            add_packet_to_fingerprint(srcMAC, dstMAC, mac_counter, mac_capture, finger_num, col)
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
            elif tcp_port == 443 or ip.data.sport == 443:  # f10: HTTPS
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
            elif udp_port == 69:
                col[25, 0] = 1  # f26: TFTP
            elif udp_port == 80:
                col[26, 0] = 1  # f27: QUIC

        # Protocols over both TCP and UDP
        try:
            trans_port = ip.data.dport
            if trans_port == 53:
                col[13, 0] = 1  # f14: DNS
            elif trans_port == 1883 or trans_port == 8883:
                col[23, 0] = 1  # f24: MQTT
            elif trans_port == 3478 or trans_port == 5349:
                col[27, 0] = 1  # f28: STUN
        except AttributeError:
            add_packet_to_fingerprint(srcMAC, dstMAC, mac_counter, mac_capture, finger_num, col)
            continue

        # for each packet
        add_packet_to_fingerprint(srcMAC, dstMAC, mac_counter, mac_capture, finger_num, col)
