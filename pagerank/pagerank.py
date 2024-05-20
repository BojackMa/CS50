import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        # 使用 with 语句打开文件。os.path.join(directory, filename) 构造了文件的完整路径。这个语句确保文件最终会被正确关闭。
        with open(os.path.join(directory, filename)) as f:
            # 读取文件的全部内容到字符串 contents 中。
            contents = f.read()
            # 使用正则表达式提取页面中所有 <a> 标签的 href 属性值。这里的正则表达式寻找形如 <a href="URL"> 的标签，并捕获 href 后的 URL 部分。
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            # 将提取到的链接放入一个集合中，并从集合中移除指向自身的链接（如果有的话）。这个步骤防止页面自链接。
            # set(links) - {filename}: 这个表达式执行集合的差集操作。差集意味着从 set(links) 中移除任何在 {filename} 集合中出现的元素。在这个特定的场景中，如果页面 filename 中含有指向自己的链接（即，如果页面有形如 <a href="filename.html"> 的链接），这个链接将被从链接集合中移除。这样做的原因是避免页面在分析自身链接时引入循环或不必要的自引用。
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    # 下面这一步是去掉死链，去掉那些不存在的页面，就是有一些会链接到不存在于文件的页面，比如说一个瞎编的网站，那么这样的信息就要被丢掉
    # 4.html需要被丢掉，因为没有4.html这个文件
    # {
    #     "1.html": {"2.html", "3.html"},
    #     "2.html": {"3.html"},
    #     "3.html": {"2.html", "4.html"}  # 注意这里有一个 "4.html"，但它不在字典键中
    # }

    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages # if link in pages 这个表达式确实是在检查 link 是否存在于 pages 这个字典的键中
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # 表示在给定页面语料库、当前页面和阻尼因子的情况下，随机冲浪者接下来将访问哪个页面的概率分布。大白话就是转移模型，转移到哪里

    # 初始化概率分布字典
    prob_dist = {}
    # 获取当前页面的链接列表
    links = corpus[page]

    # 如果当前页面有外链集合
    if links:
        # 遍历每个页面，初始化概率分布
        for link in corpus:
            prob_dist[link] = (1 - damping_factor) / len(corpus)

        # 对当前页面的每个外链增加额外的概率
        for link in links:
            prob_dist[link] += damping_factor / len(links)

    # 如果当前页面没有外链，视为链接到所有页面
    else:
        # 每个页面的概率均为1除以页面总数
        for link in corpus:
            prob_dist[link] = 1 / len(corpus)

    return prob_dist





def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # 初始化每个页面的访问计数
    page_rank = {page: 0 for page in corpus.keys()}

    # 从语料库的页面中随机选择一个页面作为起始页面
    current_page = random.choice(list(corpus.keys()))

    # 对起始页面计数加1
    page_rank[current_page] += 1

    # 似乎不用特殊的库也能完成任务
    # 进行n-1次采样
    # 下面的代码没有使用transition_model函数，只用了random，先用random，用一定概率判断是确定是选择链接还是选择任意
    # 如果是选择链接，则在链接中再次random选择
    # 如果是选择任意，则在任意页面中random选择
    # 这样就不是用transition_model函数，感觉是对的
    for _ in range(1, n):
        # 第一种
        if random.random() < damping_factor and corpus[current_page]:
            # 以阻尼因子的概率选择当前页面的一个链接
            current_page = random.choice(list(corpus[current_page]))
        else:
            # 以1 - 阻尼因子的概率随机选择任意一个页面
            current_page = random.choice(list(corpus.keys()))
        # 第二种
        # # 如果使用，则是下面这两个语句
        # # 获取当前页面的转移概率模型
        # probabilities = transition_model(corpus, current_page, damping_factor)
        # # 根据概率模型选择下一个页面
        # current_page = random.choices(list(probabilities.keys()), weights=probabilities.values(), k=1)[0]

        # 更新当前页面的计数
        page_rank[current_page] += 1

    total_visits = sum(page_rank.values())
    for page in page_rank:
        page_rank[page] /= total_visits

    return page_rank

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # Number of pages in the corpus
    num_pages = len(corpus)

    # 初始化
    page_rank = {page: 1 / num_pages for page in corpus}

    # 阈值
    convergence_threshold = 0.001

    # 标记变量，用于检查是否收敛
    converged = False

    # 每个页面p通过随机跳转得到的PageRank是(1 - d) * (1 / N)
    # 每个页面p通过链接跳转得到的PageRank是d * ∑(PageRank(i)/NumLinks(i))
    # 使用上述公式计算每个页面的新 PageRank 值

    # 第一种
    while not converged:
        # 创建新的PageRank字典用于此次迭代
        new_rank = {}
        for page in corpus:
            # 计算随机跳转贡献的PageRank
            random_surf_part = (1 - damping_factor) / num_pages
            # 计算链接跳转贡献的PageRank
            link_rank_part = 0
            for other_page in corpus:
                if page in corpus[other_page]:
                    link_rank_part += page_rank[other_page] / len(corpus[other_page])
                elif len(corpus[other_page]) == 0:
                    # 如果页面i没有链接，那么跳转到p的概率就是PageRank(i)/num_pages
                    # 这个可能很难理解，其实是必要的，因为当我们点击到一个死胡同页面（没有外链），我们不可能不动，我们一定是会到另外的页面的，所以此时相当于要漫游到所有页面
                    # 此时下一步应该是漫游到所有页面，其他页面从他这里获得的概率就是PageRank(i)/num_pages
                    # 因此，当分两种情况考虑时，第一种通过随机跳转得到的概率贡献为PageRank(i)/num_pages，如果不从这里获得，那就相当于默认随即跳转时，不会从这里跳转到页面p，这显然是不对的
                    # 其实，正常应该这么写，第一种情况，将所有死胡同页面单独拎出来分析，获得贡献
                    # 第二种情况，其它页面就按公式算，但是这样就不够优雅
                    link_rank_part += page_rank[other_page] / num_pages
            # 综合计算新的PageRank
            new_rank[page] = random_surf_part + damping_factor * link_rank_part
    # 第二种
    # while not converged:
    #     # 创建新的PageRank字典用于此次迭代
    #     new_rank = {page: 0 for page in corpus}
    #
    #     # 计算每个页面的新PageRank
    #     for page in corpus:
    #         # 累加从其他页面i（包括本页面）到当前页面p的PageRank贡献
    #         for other_page in corpus:
    #             # 获取other_page到所有页面的转移概率
    #             probabilities = transition_model(corpus, other_page, damping_factor)
    #
    #             # 累加从other_page到page的PageRank贡献
    #             if page in probabilities:
    #                  new_rank[page] += page_rank[other_page] * probabilities[page]

        # 检查是否收敛
        converged = True
        for page in corpus:
            if abs(new_rank[page] - page_rank[page]) > convergence_threshold:
                converged = False
                break

        # 更新PageRank为最新计算结果
        page_rank = new_rank

    return page_rank


if __name__ == "__main__":
    main()
