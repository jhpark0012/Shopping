from collections import deque
from PIL import Image
from streamlit_option_menu import option_menu
import pandas as pd
import copy
import streamlit as st
import datetime


class Mart:
    def __init__(self, data, x, y):
        self.dataset = data  # 데이터 불러오기

        self.row = 38
        self.col = 40  # A마트의 행, 열 크기
        self.graph = [[0] * self.col for _ in range(self.row)]  # A마트 전체 크기

        self.store_num_and_product = x  # 가게 번호와 구매할 상품 목록을 딕셔너리로 저장
        self.input_store_num = y  # 입력된 가게 번호만 담은 리스트

        self.all_destination_coord = []  # 원하는 가게의 목적지 좌표 리스트
        self.last_purchase_coord = deque()  # 맨마지막에 구매하는 아이스크림 or 쌀  목적지 좌표 리스트
        self.dst_coord = []  # 실제 최단거리를 계산할 때 사용하는 좌표 리스트 (아이스크림 + 쌀 제외)

        self.path = []  # 장보는 경로 담는 리스트,
        self.path_length = 0  # 최단 경로 길이

        self.start = (37, 20)  # 시작 지점
        self.counter_num = []  # 마지막으로 지나는 카운터 번호 리스트

        self.store_num_by_path = []  # path에 맞게 가게 번호 재정렬

        self.result = None  # 출력 결과 데이터 프레임
        self.close_counter = None # 가장 가까운 계산대


    # 그래프 그리는 함수
    def make_graph(self, row=38, col=40):

        # 테두리 (열) *로 그리기
        for i in range(col):
            self.graph[0][i] = '*'
            self.graph[-1][i] = '*'

        # 테두리 (행) *로 그리기
        for i in range(row):
            self.graph[i][0] = '*'
            self.graph[i][-1] = '*'

        # 각 구역 *로 칠하기
        for idx in range(1, len(self.dataset)):
            x, y = self.dataset.at[idx, 'left_up'], self.dataset.at[idx, 'right_bottom']

            for i in range(len(x)):
                for j in range(x[i][0], y[i][0] + 1):
                    for k in range(x[i][1], y[i][1] + 1):
                        self.graph[j][k] = '*'

        # 시작점은 S로
        for x, y in self.dataset.at[0, 'destination']:
            self.graph[x][y] = 'S'

        # 박스 있는 곳도 별표 처리
        for i in range(32, 38):
            for j in range(8):
                self.graph[i][j] = '*'

    # 모든 목적지 좌표 리턴하는 함수
    # 각 구역의 번호 리스트를 입력받아 그 번호의 목적지 좌표 리스트를 추가
    # 아이스크림과 쌀은 맨마지막에 구매하므로 따로 리스트(덱)를 만들어 그곳에 추가
    def check_all_destination(self):

        self.make_graph()  # 그래프 그리기

        for num in self.input_store_num:
            dst = self.dataset.at[num, 'destination']
            if num == 25:  # 아이스크림 구매 ( 아이스크림 > 쌀 순서로 구매)
                self.last_purchase_coord.appendleft(dst)  # 우선순위를 위헤 덱 맨 처음에 추가

            elif num == 26:  # 쌀 구매
                self.last_purchase_coord.append(dst)
            else:
                self.all_destination_coord.append(dst)

        self.last_purchase_coord = list(self.last_purchase_coord)

    # 목적지 좌표를 받아 중복되는 좌표가 있으면 그 좌표를 한 번만 사용하도록 좌표를 정리하는 함수
    def find_overlap_coord(self):

        check = [False] * len(self.all_destination_coord)  # 중복이 있었는지 체크하는 리스트

        for i in range(len(self.all_destination_coord) - 1):
            flag = False
            if not check[i]:
                tmp = set(self.all_destination_coord[i])

                for j in range(i + 1, len(self.all_destination_coord)):
                    tmp2 = set(self.all_destination_coord[j])

                    if tmp2 & tmp:
                        check[i], check[j] = True, True
                        flag = True
                        self.dst_coord.append(list(tmp & tmp2))
                        break

                if not flag:
                    self.dst_coord.append(self.all_destination_coord[i])

        if not check[-1]:  # 마지막 요소는 중복이 없으면 그대로 추가
            self.dst_coord.append(self.all_destination_coord[-1])

    # 시작점에서 나머지 목적지까지의 거리를 BFS 탐색으로 계산
    # 목적지 좌표와 시적점을 매개변수로 받음
    def bfs(self, end, start):

        dx = (0, 0, 1, -1, -1, -1, 1, 1)
        dy = (1, -1, 0, 0, -1, 1, -1, 1)

        candidate_length = []

        # 하나의 목적지에도 최대 두 개까지의 목적지 좌표가 있을 수 있으므로
        # 그 두 좌표 중 시작점과 가까운 좌표가 어딘지 계산
        # 이 좌표를 후보x, 후보y로 명명
        for candidate_x, candidate_y in end:

            tmp_graph = copy.deepcopy(self.graph)

            start_x, start_y = start  # 시작점 언패킹

            tmp_graph[start_x][start_y] = 0

            qu = deque()
            qu.append((start_x, start_y))

            while qu:
                x, y = qu.popleft()

                if x == candidate_x and y == candidate_y:
                    candidate_length.append(tmp_graph[x][y])  # 거리 추가
                    break

                # 8방향으로 탐색
                for i in range(8):
                    nx, ny = x + dx[i], y + dy[i]
                    if 0 <= nx < self.row and 0 <= ny < self.col and tmp_graph[nx][ny] == 0:
                        tmp_graph[nx][ny] = tmp_graph[x][y] + 1
                        qu.append((nx, ny))

        return candidate_length

        # 시작점에서 목적지까지의 최단 거리를 계산해서 가장 가까운 지점의 좌표를 담는 함수 (by 그리디) )

    def start_to_dst_coord(self):

        # 목적지를 방문했는지 체크 여부
        visited, iter_n = [0] * len(self.dst_coord), len(self.dst_coord)

        while sum(visited) < iter_n:  # 모든 목적지를 방문할 때 까지
            res = []

            for end in self.dst_coord:
                if end == 0:
                    res.append([(-1, -1), 10 ** 4])
                    continue

                length = 0
                candidate_length = self.bfs(end, self.start)

                idx = candidate_length.index(min(candidate_length))

                res.append([end[idx], min(candidate_length)])

            closest = sorted(res, key=lambda x: x[1])[0]  # 가장 가까운 (좌표, 거리)

            visited[res.index(closest)] = 1  # 방문 체크
            self.start = closest[0]  # 가장 가까운 지점을 시작점으로 다시 세팅

            self.path.append(self.start)  # 시작점 추가
            self.path_length += closest[1]  # 가장 가까운 거리 더하기

            self.dst_coord[res.index(closest)] = 0  # 거리를 계산하지 않도록 하기 위해 0으로 표시

    # 마지막 지점에서 아이스크림 또는 살 구매
    def icecream_and_rice(self):

        for end in self.last_purchase_coord:
            res = []

            candidate_length = self.bfs(end, self.start)

            idx = candidate_length.index(min(candidate_length))
            res.append([end[idx], min(candidate_length)])

            min_coord = sorted(res, key=lambda x: x[1])

            self.path.append(min_coord[0][0])
            self.start = min_coord[0][0]
            self.path_length += min_coord[0][1]

    # 마지막 구매 지점에서 카운터까지 거리 계산
    def move_to_counter(self):

        # 카운터 좌표
        goal_idx = [(31, 10), (31, 14), (31, 18)]

        # 카운터 번호
        goal_num = [48, 49, 50]
        goal_length = self.bfs(goal_idx, self.start)

        self.path_length += min(goal_length)
        x, y = goal_idx[goal_length.index(min(goal_length))]

        self.path.append((x, y))
        self.counter_num.append(goal_num[goal_length.index(min(goal_length))])

        return self.counter_num  # 마지막으로 지나는 카운터 번호 리턴

    # path에 맞게 가게 번호 재정렬
    def get_store_num_order(self):

        store_num_by_destination = []

        for p in self.path:
            for i in range(len(self.dataset)):
                if p in self.dataset.loc[i, 'destination']:
                    store_num_by_destination.append(i)

        diff_store_num = set(store_num_by_destination) - set(self.input_store_num)

        for x in store_num_by_destination:
            if x not in diff_store_num:
                self.store_num_by_path.append(x)

        self.store_num_by_path += self.counter_num

    # 결과를 데이터 프레임으로 만든 뒤, csv 파일로 저장
    def save_result(self):
        product_list = {'구역이름': [], '상품목록': []}

        for num in self.store_num_by_path:
            if num < 48:
                product_list['구역이름'].append(self.dataset.loc[num, "name"])
                product_list['상품목록'].append(self.store_num_and_product[num])

        self.result = pd.DataFrame(product_list)
        self.close_counter = self.dataset.loc[self.store_num_by_path[-1], "name"]

        self.result.to_csv('result.csv', index=False)

    # 출력 함수
    def print_info(self):

        result = pd.read_csv('result.csv')

        st.markdown("<h3 style='text-align: center; color: white;'>** 다음 순서대로 물건을 담으시면 됩니다 **</h3>",
                    unsafe_allow_html=True)
        st.dataframe(result, width=600,)

        st.text('---------------------------------------------------------------------------')
        st.text(f'가장 가까운 계산대: {self.close_counter}')
        st.text(f'총 이동 거리: {self.path_length} 걸음')

    def shopping(self):

        self.check_all_destination()  # 목적지 체크
        self.find_overlap_coord()  # 중복 좌표 체크

        self.start_to_dst_coord()  # 최단경로 계산

        if self.last_purchase_coord:
            self.icecream_and_rice()  # 아이스크림 or 쌀 경로 계산

        self.move_to_counter()  # 가장 가까운 카운터 경로 계산

        self.get_store_num_order()  # 번호 재정렬

        self.save_result()  # 결과 저장
        self.print_info()  # 결과 출력

def input_box_func(p_type, p_list, name):
    # add the key choices_len to the session_state
    if not "choices_len" in st.session_state:
        st.session_state["choices_len"] = 0

    with st.container():
        col_l, _, col_r = st.columns((4, 15, 3))
        with col_l:
            if st.button("Add"):
                st.session_state["choices_len"] += 1

        with col_r:
            if st.button("Remove") and st.session_state["choices_len"] > 0:
                st.session_state["choices_len"] -= 1
                st.session_state.pop(f'{st.session_state["choices_len"]}')

    for x in range(st.session_state["choices_len"]):  # create many choices
        c11, c22 = st.columns(2)
        with st.container():
                p_type.append(c11.selectbox('구역 이름을 선택하세요', name, key=f'{x+100}'))
                p_list.append(c22.text_input("상품 이름을 입력하세요", key=f"{x}"))



def home_page():
    ## Title
    st.markdown("<h1 style='text-align: center; color: yellow;'>Shortest Path Shopping</h1>", unsafe_allow_html=True)
    st.image(Image.open('structure.png'))

    ## Header
    st.markdown(f"<h3 style='text-align: center; color: white;'>{datetime.date.today()} 오늘 살 것</h3>", unsafe_allow_html=True)

    ## 데이터 (피클 파일)
    data = pd.read_pickle('dataset_kor.pkl')
    name = list(data['name'])[:48]

    # 상품 유형과 목록 리스트
    p_type = []
    p_list = []
    p_type_by_num = []
    input_box_func(p_type, p_list, name)  # 장보는 함수 호출

    if st.button('Shopping!'):
        if not p_type:  # 아무것도 입력 안했을 때 에러
            st.error("구역과 상품을 입력하세요!")

        elif 'IN' in p_type: # 구역 이름에 IN 있을 때 에러
            st.error(f'{p_type.index("IN") + 1}번째 구역 이름을 다시 확인해보세요! ')

        elif '' in p_list: # 상품 이름 입력 안 했을 때 에러
            st.error(f'{p_list.index("") + 1}번째 상품 이름을 입력하세요! ')


        else:

            for p_t in p_type:
                p_type_by_num.append(data[data['name'] == p_t].index[0])

            store_num_and_product = dict()

            for num, p_l in zip(p_type_by_num, p_list):
                if num in store_num_and_product.keys():
                    store_num_and_product[num].append(p_l)
                else:
                    store_num_and_product[num] = []
                    store_num_and_product[num].append(p_l)

            obj = Mart(data, store_num_and_product, list(store_num_and_product.keys()))  # x는 딕셔너리, y는 리스트
            obj.shopping()



def use_page():
    st.markdown("<h1 style='text-align: center; color: yellow;'>How to Use</h1>", unsafe_allow_html=True)

    # st.image(Image.open('structure.png'))

    st.write("""
            ###
            ##### 1. 적어둔 구매 리스트를 꺼내세요
            ##### 
            ##### 2. Add 버튼을 눌러 구역 이름과 상품 이름을 추가하세요
            ##### 
            ##### 3. 구역 이름을 선택하고 상품 이름을 입력하세요
                - 1. 구역 이름을 모를 경우 '참고' 에서 확인하세요(추가 예정)
                - 2. 한 구역에 모든 상품을 입력하지 않아도 됩니다
                - 3. 약간의 로딩이 있을 수 있으니 침착히 기다리세요
            ##### 
            ##### 4. 모든 입력이 완료 됐다면 비어있는 칸은 Remove 버튼으로 지우세요
            ##### 
            ##### 5. Shopping! 버튼을 눌러 가장 빠르게 장을 볼 수 있는 방법을 확인하세요
                - 1. 에러가 나면 에러 내용을 잘 읽고 수정하세요
                - 2. 쌀과 아이스크림은 맨 마지막에 구매하도록 설정했습니다
                - 3. 아이스크림 > 쌀 순으로 구매하도록 설정했습니다 
            #####
            ##### 6. 거리 계산 방법이 궁금하면 개발자에게 물어보세요
            """)


def note_page():
    st.markdown("<h1 style='text-align: center; color: yellow;'>Note</h1>", unsafe_allow_html=True)

    st.image(Image.open('structure.png'))

    with st.container():
        c1, c2 = st.columns(2)

        c1.image(Image.open('s_1_to_5.png'))
        c2.image(Image.open('s_6_to_10.png'))

        # c2.image(Image.open('s_1_to_5_long.png'))
        # c3.image(Image.open('s_6_to_10_long.png'))



if __name__ == "__main__":
    # with st.sidebar:

    selected = option_menu(
        menu_title = 'Main Menu',
        options = ['홈', '사용법', '참고'],
        icons = ['house', 'question-circle', 'exclamation-circle'],
        menu_icon = 'cart',
        default_index = 0,
        orientation='horizontal',
    )

    if selected == '홈':
        home_page()

    if selected == '사용법':
        use_page()

    if selected == '참고':
        note_page()