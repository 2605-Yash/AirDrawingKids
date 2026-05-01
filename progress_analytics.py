import statistics

def best_score(profile): return max(profile.final_hist) if profile.final_hist else 0
def average_score(profile): return statistics.mean(profile.final_hist) if profile.final_hist else 0

def improvement_percent(profile):
    hist=list(profile.final_hist)
    if len(hist)<6:return 0.0
    first=statistics.mean(hist[:3]); last=statistics.mean(hist[-3:])
    if first==0:return 0.0
    return ((last-first)/first)*100.0

def consistency(profile):
    hist=list(profile.final_hist)
    if len(hist)<4:return 0.0
    std=statistics.pstdev(hist)
    return max(0,100-std*2.2)

def session_summary(profile):
    return {
        "average":round(average_score(profile),1),
        "best":round(best_score(profile),1),
        "improvement":round(improvement_percent(profile),1),
        "consistency":round(consistency(profile),1),
        "weakness":profile.weakest_area(),
        "summary":"Adaptive learning active"
    }
