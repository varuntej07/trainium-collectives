"""
The Alpha-Beta Model helps us account for the cost of communication when analyzing the time of an algorithm
Alpha (α) - represents the network latency cost of the message (time/message). 
            It is a constant factor for each message sent.
Beta (β) - represents the inverse bandwidth cost to send each word of the message (time/word). 
            It grows linearly with the size of the message.

"""

def step_time(alpha: float, chunk_bytes: int, bandwidth_bytes_s: float) -> float:
    """
    For communication, before any useful byte lands in other processes, you pay for: 
    software initiating the transfer, the handshake, and the signal physically propagating down the wire. 
    Whether you send 1 bit or 1 byte or 1 MB, you pay this once. Call it alpha (seconds).

    Once the pipe is flowing, the bytes stream through at the link's bandwidth B (bytes/sec). 
    To push 'n' bytes (chunks) through a pipe that moves 'B' bytes every second takes n / B seconds. 

    so, step_time = alpha + n * beta,   # where beta = 1 / B (seconds/byte) is the time to send one byte.
    """
    if bandwidth_bytes_s <= 0:
        raise ValueError(f"bandwidth_bytes_s must be > 0, got {bandwidth_bytes_s}")
    
    beta = 1 / bandwidth_bytes_s
    return alpha + chunk_bytes * beta


def bandwidth_util(useful_bytes: int, bandwidth_bytes_s: float, total_time_s: float) -> float:
    """
    Once we know how long a whole collective took, we want to know how well it used the link.
    The link can move 'B' bytes every second at full speed, so over 'total_time_s' seconds the most
    it could ever carry is B * total_time_s bytes. That is the peak - the best case if the link never sat idle.

    But only useful_bytes of that was payload that actually mattered. So the fraction of the
    link's peak we actually achieved is:

        bandwidth_util = useful_bytes / (B * total_time_s)

    It normally sits in [0, 1]. The gap below 1 is the time we spent paying latency (alpha) and
    waiting instead of streaming bytes, so a low number says "latency is eating us". If it ever
    comes out above 1, we "moved more than the link can physically carry", which is impossible,
    so, it means our assumed peak B is too low and needs recalibrating against measured numbers.
    That is a finding, not a bug.
    """

    if bandwidth_bytes_s <= 0:
        raise ValueError(f"bandwidth_bytes_s must be > 0, got {bandwidth_bytes_s}")
    if total_time_s <= 0:
        raise ValueError(f"total_time_s must be > 0, got {total_time_s}")

    return useful_bytes / (bandwidth_bytes_s * total_time_s)
