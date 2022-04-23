def return_similarity(a: str,b: str):
    size = len(a) if len(a) > len(b) else len(b)
    a = a.ljust(size)
    b = b.ljust(size)
    equals = 0
    for i in range(size):
        if(a[i]==b[i]):
            equals+=1
    
    return {"equals": equals, "diff": size-equals,"similarity": equals/size}