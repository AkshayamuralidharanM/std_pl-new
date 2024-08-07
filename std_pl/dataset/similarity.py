from skimage.metrics import  structural_similarity
import  cv2
def orb_sim(imag1,image2):
    orb=cv2.ORB_create()
    kp_a,desc_a=orb.detectAndCompute(imag1,None)
    kp_b,desc_b=orb.detectAndCompute(imag1,None)
    bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
    matches=bf.match(desc_a,desc_b)
    similar_regions=[i for i in matches if i.distance<50]
    if len(matches)==0:
        return 0
    return  len(similar_regions)/len(matches)    
img00=cv2.imread('D:/std_pl-new/std_pl/assest/images/img1.jpg',0)    
img01=cv2.imread('D:/std_pl-new/std_pl/assest/images/img2.jpg',0)  


def structural_sim(img1,img2):
    sim,diff=structural_similarity(img1,img2,full=True)
    return sim
orb_similarity=orb_sim(img00,img01)   
print("similarity using ORB is:",orb_similarity) 

# from skimage.transfrom import resize
# img1 = resize(img00, (img00.shape[0] // 4, img00.shape[1] // 4),
#                        anti_aliasing=True)
# img2 = resize(img01, (img01.shape[0] // 4, img01.shape[1] // 4),
#                        anti_aliasing=True)  
# ssim=structural_sim(img00,img01)
# print("Sim",ssim)

