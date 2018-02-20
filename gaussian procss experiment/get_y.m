
% find the closest point in the grid to the points pos
function [val,all_index]=get_y(pos,grid,data)   
        
    n=size(pos,1);
    val=zeros(n,1);
    all_index=zeros(n,1);
    for i=1:n
        x=pos(i,1);
        y=pos(i,2);
        distance=(grid(:,1)-x).^2+(grid(:,2)-y).^2; 
        [v,index]=min(distance);
        val(i,:)=data(index);
        all_index(i,:)=index;
    end
    
end