function [ res ] = ClusterFaces( list)
   N = size(list,1);
   mset = -ones(N,1);

   for i = 1:N
       for j = i+1:N
          if  GetDist(list(i,:),list(j,:)) > 0.5
              [mset] = merge(mset,i,j);
          end
       end
   end
   
   ind = mset<0 ;
   
   s = zeros(size(list));
   
   for i = 1:N
      [r ,mset] = find(mset,i) ;
      s(r,:) =  s(r,:) + list(i,:);
   end
   
  for i = 1:N
     s(i,:) = s(i,:) / (-mset(i));
  end
   
     res = s(ind,:);
   


end


function [r ,mset] = find(mset,no)
   if (mset(no)<0)
       r = no;
       return ;
   end;
   [r,mset] = find(mset,mset(no));
   mset(no) = r;
end

function [mset] = merge(mset,a,b)
    [r1 ,mset] = find(mset,a);
    [r2 ,mset] = find(mset,b);
    
    if r1 == r2
        return;
    end
    
    mset(r1) =  mset(r1) + mset(r2);
    mset(r2) = r1;
end

function [minp] = GetDist(ra,rb)

    pa = getpoints(ra);
    pb = getpoints(rb);
    
    lx = [ pa(1,1) pa(2,1)  pb(1,1) pb(2,1) ];
    ly = [ pa(2,2) pa(3,2)  pb(2,2) pb(3,2) ];
    
    dxa = pa(2,1) - pa(1,1);
    dxb = pb(2,1) - pb(1,1);
    
    dya = pa(3,2) - pa(2,2);
    dyb = pb(3,2) - pb(2,2);
    
    if (max(lx)-min(lx) <= dxa +dxb) && (max(ly)-min(ly) <= dya +dyb)
        
        area = (dxa +dxb - max(lx) + min(lx))*(dya +dyb - max(ly) + min(ly));
        
        minp = min([area/dxa/dya  area/dxb/dyb]);
        return;
     end
    
    minp = 0;

end

function [points] = getpoints(list)
    points = zeros(4,2);
    points(1,:) = [list(1) - list(3)/2 , list(2) - list(4)/2];
    points(2,:) =  points(1,:) + [ list(3) 0];
    points(3,:) =  points(2,:) + [ 0 list(4)];
    points(4,:) =  points(1,:) + [ 0 list(4)];

end
