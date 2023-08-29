# Shopping 
[1st Toy Project] Shortest Shopping Path Web

## Period
22.01 ~ 22.02

## Background
우리가 장을 볼 때, 장 볼거리를 미리 적어놓고 마트에 간다. 하지만 대부분 메모를 할 때 상품 카테고리별로 적지 않고 생각나는대로 적는다.

그래서 마트에 도착하면 메모지를 계속해서 처음부터 끝까지 확인을 하거나 왔던 곳을 다시 와서 상품을 고르는 불편함이 발생한다.

이러한 문제를 해결하고자 최근에 배운 BFS 알고리즘과 streamlit을 활용하여 최단 거리의 동선을 알려주는 웹을 만들고자 한다.


## Data
- **structure.png**
  
  마트의 구조를 그림으로 나타낸 data
  
- **s_1_to_5.png, s_6_to_10.png**

  10개의 section을 나타낸 data

- **dataset_kor.csv, dataset_kor.pkl**

  structure.png, s_1_to_5.png, s_6_to_10.png 를 표로 나타낸 data

- **svg data**

  web 이모티콘 data


## Expected Outcomes
효율적인 동선으로 장을 빠르게 볼 수 있게 되어 불필요한 시간 낭비를 줄일 수 있게 되었고, 

마트에 오래 머물러 있지 않아 기존에 적은 상품 외에 불필요한 상품을 구매하는 행동을 안 할 수 있어 경제적인 소비 습관을 할 수 있게 될 것으로 기대된다.

## Future Work
입력이 주어질 때 마다 각 구간 별 거리를 계산하는 것이 아니라 그래프 자료구조를 사용하여 구간별 거리를 미리 저장해 놓고 있다면 더 빠르게 동선을 계산할 수 있을 것이다.
