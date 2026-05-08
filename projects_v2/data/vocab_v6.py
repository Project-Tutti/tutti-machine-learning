def build_v6_vocab():
    """
    V6 vocab 빌드. 
    스펙 문서에 정의된 순서대로 토큰을 등록하여 총 645개의 토큰 ID를 생성합니다.
    토큰 ID는 등록 순서대로 0부터 연속 할당됩니다.
    
    Returns:
        vocab (dict): {token_str: token_id} 형태의 딕셔너리
    """
    vocab = {}
    def add(token):
        if token not in vocab:
            vocab[token] = len(vocab)
    
    # 1. FIM 마커 (3종)
    for t in ["<PRE>", "<SUF>", "<MID>"]:
        add(t)
    
    # 2. 구조 토큰 (7종)
    for t in ["PAD", "BOS", "EOS", "PIECE_START", "PIECE_END", "BAR_START", "BAR_END"]:
        add(t)
    
    # 3. 장르 (5종)
    for g in ["POP", "ROCK", "ELECTRONIC", "CLASSICAL", "OTHER"]:
        add(f"GENRE_{g}")
    
    # 4. 키 (12 × 2 + NONE = 25종)
    for r in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]:
        for m in [":maj", ":min"]:
            add(f"KEY_{r}{m}")
    add("KEY_NONE")
    
    # 5. 박자 (6종)
    for m in ["4:4", "3:4", "2:4", "6:8", "12:8", "OTHER"]:
        add(f"METER_{m}")
    
    # 6. TARGET_INST (17종)
    for p in [40, 41, 42, 43, 56, 57, 58, 60, 64, 65, 66, 67, 68, 69, 70, 71, 73]:
        add(f"TARGET_{p}")
    
    # 7. DENSITY (1~5 = 5종)
    for d in range(1, 6):
        add(f"DENSITY_{d}")
    
    # 8. INST (0~128, 128=드럼 = 129종)
    for i in range(129):
        add(f"INST={i}")
    
    # 9. TIME (0~95 = 96종)
    for t in range(96):
        add(f"TIME={t}")
    
    # 10. PITCH (0~127 = 128종)
    for p in range(128):
        add(f"PITCH={p}")
    
    # 11. DUR (1~192 = 192종)
    for d in range(1, 193):
        add(f"DUR={d}")
    
    # 12. VEL (0~31 = 32종)
    for v in range(32):
        add(f"VEL={v}")
    
    return vocab

def get_inverse_vocab(vocab):
    """
    {token_id: token_str} 형태의 inverse_vocab을 생성합니다.
    디코딩이나 디버깅 시 유용하게 사용됩니다.
    """
    return {v: k for k, v in vocab.items()}

if __name__ == "__main__":
    vocab = build_v6_vocab()
    print(f"Total vocabulary size: {len(vocab)}")
    assert len(vocab) == 645, f"Expected 645 tokens, got {len(vocab)}"
    print("Vocab built successfully! First 10 tokens:")
    for i, (k, v) in enumerate(vocab.items()):
        if i < 10:
            print(f"  {k}: {v}")
