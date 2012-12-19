function iassign = ruleblkassign(mat, lens, ppp)
  matp = mat(:,ppp);
  size1 = size(mat,1);
  iassign = zeros(size1,3);
  [dmy is1] = sortrows(matp);
  for i = 1:lens(ppp(1))
    istart = floor((i-1)*size1/lens(ppp(1)))+1;
    iend = floor(i*size1/lens(ppp(1)));    
    ip1 = is1(istart:iend,1);
    iassign(ip1,ppp(1)) = i;
    mat2 = matp(ip1,2:3);
    [dmy is2] = sortrows(mat2);
    size2 = length(is2);
    for j = 1:lens(ppp(2))
      jstart = floor((j-1)*size2/lens(ppp(2)))+1;
      jend = floor(j*size2/lens(ppp(2)));    
      ip2 = is2(jstart:jend);
      iassign(ip1(ip2),ppp(2)) = j;
      mat3 = mat2(ip2,2);
      [dmy is3] = sort(mat3);
      size3 = length(is3);
      for k = 1:lens(ppp(3))
        kstart = floor((k-1)*size3/lens(ppp(3)))+1;
        kend = floor(k*size3/lens(ppp(3)));    
        ip3 = is3(kstart:kend);
        iassign(ip1(ip2(ip3)),ppp(3)) = k;
      end
    end
  end
        
      
    