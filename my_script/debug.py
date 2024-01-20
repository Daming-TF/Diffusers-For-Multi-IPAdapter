
for i in range(10):
    try:
        assert i <5,f"i mus < 5, but i = {i}"
        print(i)
    except Exception as e:
        print(f"{e}")
        continue
